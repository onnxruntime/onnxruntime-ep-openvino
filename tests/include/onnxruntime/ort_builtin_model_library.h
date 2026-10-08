// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <variant>
#include "data_utils.h"
#include "half.h"
#include "network.h"

namespace ovep_tests {

///////////////////////////////////////////////////////////////////////
// Comparison strategy for output validation
enum class comparison_strategy {
  exact_match,       // Byte-for-byte comparison
  cosine_similarity  // Cosine similarity with configurable threshold
};

///////////////////////////////////////////////////////////////////////
// Element type for tensor data
enum class element_type {
  float32,
  float16,
  int8,
  uint8,
  int32,
  int64
};

///////////////////////////////////////////////////////////////////////
// Model I/O data type
using io_byte_data_t = std::map<std::string, std::vector<std::byte> >;

///////////////////////////////////////////////////////////////////////
// Helpers for inline test data (converts typed values to byte vectors)
inline std::vector<std::byte> floats_to_bytes(std::initializer_list<float> values) {
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), std::data(values), bytes.size());
  return bytes;
}

inline std::vector<std::byte> bool_to_bytes(bool value) {
  return {static_cast<std::byte>(value ? 1 : 0)};
}

template <typename T>
using io_data_t = std::map<std::string, std::vector<T> >;

///////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<std::byte> to_byte_vec(const std::vector<T>& data) {
  const auto* start = reinterpret_cast<const std::byte*>(data.data());
  const auto* end = start + (data.size() * sizeof(T));
  return std::vector<std::byte>(start, end);
}

// Deterministic ramp input (i % 7) / 7, as float bytes. Lets large inputs be defined in code rather
// than stored as binary test data.
inline std::vector<std::byte> ramp_floats_to_bytes(size_t count) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>(i % 7) / 7.0f;
  }
  return to_byte_vec(values);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
concept ByteContainer = requires(T t) {
  {
    t.data()
  } -> std::convertible_to<const std::byte*>;
  {
    t.size()
  } -> std::convertible_to<size_t>;
};

///////////////////////////////////////////////////////////////////////
template <typename T, ByteContainer ByteVec>
auto get_as(ByteVec&& data) {
  if (data.size() % sizeof(T) != 0) {
    throw std::runtime_error("Data size is not a multiple of element size.");
  }
  size_t count = data.size() / sizeof(T);

  using byte_t = std::remove_reference_t<decltype(data[0])>;
  using result_t = std::conditional_t<std::is_const_v<byte_t>, const T, T>;
  return std::span<result_t>(reinterpret_cast<result_t*>(data.data()), count);
}

///////////////////////////////////////////////////////////////////////
// Dispatch function based on element type
template <typename Func>
auto dispatch_by_element_type(element_type elem_type, Func&& func) {
  switch (elem_type) {
    case element_type::float32:
      return func.template operator()<float>();
    case element_type::float16:
      return func.template operator()<half>();
    case element_type::int8:
      return func.template operator()<int8_t>();
    case element_type::uint8:
      return func.template operator()<uint8_t>();
    case element_type::int32:
      return func.template operator()<int32_t>();
    case element_type::int64:
      return func.template operator()<int64_t>();
    default:
      OVEP_TESTS_THROW("Unsupported element type.");
  }
}

///////////////////////////////////////////////////////////////////////
// Forward declaration
template <typename fixture_t>
class builtin_model_session;

///////////////////////////////////////////////////////////////////////
// Model metadata and session factory
template <typename fixture_t>
class builtin_model_info {
 public:
  using device_t = typename fixture_t::device_wrapper_t;
  using graph_t = typename fixture_t::graph_wrapper_t;
  using pipeline_t = typename fixture_t::pipeline_wrapper_t;
  using session_options_t = typename fixture_t::session_options_wrapper_t;
  using builtin_session_t = builtin_model_session<fixture_t>;

  ///////////////////////////////////////////////////////////////////////
  // Tensor metadata combining source (file or in-memory), element type, and comparison config
  struct tensor_metadata {
    std::variant<std::filesystem::path, std::vector<std::byte> > source;
    element_type elem_type = element_type::float32;
    comparison_strategy strategy = comparison_strategy::cosine_similarity;
    double tolerance = 0.97;

    bool is_file() const { return std::holds_alternative<std::filesystem::path>(source); }
    bool is_memory() const { return std::holds_alternative<std::vector<std::byte> >(source); }
  };

  std::string name_;
  std::filesystem::path model_path_;
  std::filesystem::path external_data_path_;  // Empty if no external data (e.g., .bin file)

  // Input/output tensor metadata (maps tensor name to metadata)
  std::map<std::string, tensor_metadata> inputs_;
  std::map<std::string, tensor_metadata> outputs_;

  // Model characteristics
  bool has_dynamic_shapes_ = false;
  bool cpu_fallback_required_ = false;

  ///////////////////////////////////////////////////////////////////////
  std::unique_ptr<builtin_session_t> create_session(const std::string& device_name) const {
    return std::make_unique<builtin_session_t>(*this, session_options_t(device_t(device_name)));
  }

  ///////////////////////////////////////////////////////////////////////
  std::unique_ptr<builtin_session_t> create_session(session_options_t session_options) const {
    return std::make_unique<builtin_session_t>(*this, std::move(session_options));
  }

  ///////////////////////////////////////////////////////////////////////
  std::unique_ptr<builtin_session_t> create_session(const std::string& device_name, network_t network_config)
      const {
    return std::make_unique<builtin_session_t>(*this, device_name, std::move(network_config));
  }

  ///////////////////////////////////////////////////////////////////////
  static Ort::AllocatedStringPtr GetCompatibilityString(const std::filesystem::path& model_path,
                                                        const char* ep_type = "OpenVINOExecutionProvider") {
    Ort::AllocatorWithDefaultOptions allocator;
    return Ort::GetCompatibilityInfoFromModelAllocated(model_path.c_str(), ep_type, allocator);
  }

  ///////////////////////////////////////////////////////////////////////
  std::vector<uint8_t> load_model_buffer() const {
    std::vector<uint8_t> buffer;
    data_utils::read_binary_file(model_path_, buffer);
    return buffer;
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T = std::byte>
  io_data_t<T> load_input_data() const {
    if (inputs_.empty()) {
      throw std::runtime_error("No input data available for model: " + name_);
    }

    io_data_t<T> input_data;
    for (const auto& [input_name, metadata] : inputs_) {
      if (metadata.is_file()) {
        input_data[input_name] =
            data_utils::read_binary_file<T>(std::get<std::filesystem::path>(metadata.source));
      } else {
        const auto& byte_vec = std::get<std::vector<std::byte> >(metadata.source);
        auto span = get_as<T>(byte_vec);
        input_data[input_name] = std::vector<T>(span.begin(), span.end());
      }
    }
    return input_data;
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T = std::byte>
  io_data_t<T> load_expected_output_data() const {
    if (outputs_.empty()) {
      throw std::runtime_error("No output data available for model: " + name_);
    }

    io_data_t<T> output_data;
    for (const auto& [name, metadata] : outputs_) {
      if (metadata.is_file()) {
        output_data[name] =
            data_utils::read_binary_file<T>(std::get<std::filesystem::path>(metadata.source));
      } else {
        const auto& byte_vec = std::get<std::vector<std::byte> >(metadata.source);
        auto span = get_as<T>(byte_vec);
        output_data[name] = std::vector<T>(span.begin(), span.end());
      }
    }
    return output_data;
  }

  ///////////////////////////////////////////////////////////////////////
  // Validate actual outputs against expected outputs using configured comparison strategies
  void validate_outputs(const io_byte_data_t& actual_outputs) const {
    auto expected_outputs = load_expected_output_data<std::byte>();
    validate_outputs(actual_outputs, expected_outputs);
  }

  ///////////////////////////////////////////////////////////////////////
  // Validate with pre-loaded expected outputs (avoids reloading from disk)
  void validate_outputs(const io_byte_data_t& actual_outputs, const io_byte_data_t& expected_outputs) const {
    // Validate each output using its configured comparison strategy
    for (const auto& [name, metadata] : outputs_) {
      auto actual_it = actual_outputs.find(name);
      auto expected_it = expected_outputs.find(name);

      if (actual_it == actual_outputs.end()) {
        OVEP_TESTS_THROW("Missing actual output: %s", name.c_str());
      }
      if (expected_it == expected_outputs.end()) {
        OVEP_TESTS_THROW("Missing expected output: %s", name.c_str());
      }

      const auto& actual = actual_it->second;
      const auto& expected = expected_it->second;

      bool passed = false;
      switch (metadata.strategy) {
        case comparison_strategy::exact_match:
          passed = std::equal(expected.begin(), expected.end(), actual.begin(), actual.end());
          if (!passed) {
            OVEP_TESTS_LOG_WARN << "Exact match failed for output '" << name << "'";
          }
          break;

        case comparison_strategy::cosine_similarity: {
          double similarity = dispatch_by_element_type(
              metadata.elem_type,
              [&expected, &actual]<typename T>() {
                auto expected_span = get_as<const T>(expected);
                auto actual_span = get_as<const T>(actual);
                return data_utils::cosine_similarity(actual_span, expected_span);
              });

          passed = similarity >= metadata.tolerance;
          if (!passed) {
            OVEP_TESTS_LOG_WARN << "Cosine similarity for output '" << name
                                << "' is below tolerance: " << similarity << " < " << metadata.tolerance;
          }
          break;
        }

        default:
          OVEP_TESTS_THROW("Unsupported comparison strategy for tensor: %s", name.c_str());
      }

      if (!passed) {
        OVEP_TESTS_THROW("Output validation failed for tensor: %s", name.c_str());
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////
// Session that holds device, network, graph, and pipeline
template <typename fixture_t>
class builtin_model_session {
 public:
  using device_t = typename fixture_t::device_wrapper_t;
  using graph_t = typename fixture_t::graph_wrapper_t;
  using pipeline_t = typename fixture_t::pipeline_wrapper_t;
  using session_t = typename fixture_t::session_wrapper_t;
  using session_options_t = typename fixture_t::session_options_wrapper_t;
  using model_info_t = builtin_model_info<fixture_t>;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(builtin_model_session);

  ///////////////////////////////////////////////////////////////////////
  // Constructor with custom network configuration
  builtin_model_session(const model_info_t& model_info, const std::string& device_name, network_t network_config)
      : model_info_(model_info), device_(device_name), network_(std::move(network_config)), graph_(nullptr), pipeline_(nullptr) {
    if (!network_.buffer_.empty() && !model_info_.external_data_path_.empty()) {
      // Only set external data path if model is loaded from buffer and not already set
      network_.ort_session_config_.emplace(
          kOrtSessionOptionsModelExternalInitializersFileFolderPath,
          path_to_utf8(model_info_.external_data_path_.parent_path()));
    }
    graph_ = std::make_unique<graph_t>(device_, network_);
  }

  ///////////////////////////////////////////////////////////////////////
  // Constructor with custom session options
  builtin_model_session(const model_info_t& model_info, session_options_t session_options)
      : model_info_(model_info), device_(session_options.device_.name_), network_(network_t::create(model_info_.model_path_)), graph_(nullptr), pipeline_(nullptr) {
    graph_ = std::make_unique<graph_t>(network_, std::move(session_options));
  }

  ///////////////////////////////////////////////////////////////////////
  // Run inference with custom I/O
  template <typename T>
  void run(const io_data_t<T>& input_data, io_data_t<T>& output_data) const {
    if (model_info_.has_dynamic_shapes_) {
      OVEP_TESTS_THROW(
          "Built-in run() does not support models with dynamic shapes. Please use ort_session().Run() with "
          "explicit tensor shapes instead.");
    }

    get_pipeline().run(input_data, output_data);
  }

  ///////////////////////////////////////////////////////////////////////
  // Run inference with built-in inputs and validate outputs using configured comparison strategy
  void run_and_validate() const {
    auto input_data = model_info_.load_input_data();
    auto expected_output = model_info_.load_expected_output_data();

    // Create output buffers
    io_byte_data_t output_data;
    for (const auto& [name, expected] : expected_output) {
      output_data[name] = std::vector<std::byte>(expected.size());
    }

    // Run inference
    run(input_data, output_data);

    // Validate using model info
    model_info_.validate_outputs(output_data, expected_output);
  }

  ///////////////////////////////////////////////////////////////////////
  // Accessors
  device_t& device() { return device_; }
  const device_t& device() const { return device_; }

  network_t& network() { return network_; }
  const network_t& network() const { return network_; }

  graph_t& graph() { return *graph_; }
  const graph_t& graph() const { return *graph_; }

  pipeline_t& pipeline() { return get_pipeline(); }
  const pipeline_t& pipeline() const { return get_pipeline(); }

  const model_info_t& model_info() const { return model_info_; }

  session_t& ort_session() const { return *graph_->session_; }

 private:
  pipeline_t& get_pipeline() const {
    if (pipeline_ == nullptr) {
      pipeline_ = std::make_unique<pipeline_t>(device_, *graph_);
    }
    return *pipeline_;
  }

  const model_info_t& model_info_;
  mutable device_t device_;
  network_t network_;
  std::unique_ptr<graph_t> graph_;
  mutable std::unique_ptr<pipeline_t> pipeline_;
};

///////////////////////////////////////////////////////////////////////
// Built-in model library
template <typename fixture_t>
class model_library {
 public:
  using model_info_t = builtin_model_info<fixture_t>;

  inline static const std::filesystem::path kTestDataDir = "testdata";

  ///////////////////////////////////////////////////////////////////////
  // Get model info by name
  static const model_info_t& GetModel(const char* model_name) {
    static const std::map<std::string, model_info_t> models = InitializeModels();

    auto it = models.find(model_name);
    if (it == models.end()) {
      throw std::runtime_error(std::string("Model not found: ") + model_name);
    }
    return it->second;
  }

 private:
  ///////////////////////////////////////////////////////////////////////
  static std::map<std::string, model_info_t> InitializeModels() {
    const auto testdata = kTestDataDir;

    return {// mul_1: Simple multiplication model (2x3 matrix * [1,2,3,4,5,6])
            {"mul_1",
             {.name_ = "mul_1",
              .model_path_ = testdata / "mul_1.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {to_byte_vec(std::vector<float>(6, 2.f))}}},
              .outputs_ = {{"Y", {to_byte_vec(std::vector<float>{2.f, 4.f, 6.f, 8.f, 10.f, 12.f})}}},
              .has_dynamic_shapes_ = false}},

            // mul_1_dynamic: Same as mul_1 but with dynamic input shapes
            {"mul_1_dynamic",
             {.name_ = "mul_1_dynamic",
              .model_path_ = testdata / "mul_1_dynamic.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {to_byte_vec(std::vector<float>(6, 2.f))}}},
              .outputs_ = {{"Y", {to_byte_vec(std::vector<float>{2.f, 4.f, 6.f, 8.f, 10.f, 12.f})}}},
              .has_dynamic_shapes_ = true}},

            // mul_1_ep_ctx_ovir: mul_1 with EP context
            {"mul_1_ep_ctx_ovir",
             {.name_ = "mul_1_ep_ctx_ovir",
              .model_path_ = testdata / "mul_1_ep_ctx_ovir.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {to_byte_vec(std::vector<float>(6, 2.f))}}},
              .outputs_ = {{"Y", {to_byte_vec(std::vector<float>{2.f, 4.f, 6.f, 8.f, 10.f, 12.f})}}},
              .has_dynamic_shapes_ = false}},

            // mul_2_large: Larger multiplication model
            {"mul_2_large",
             {.name_ = "mul_2_large",
              .model_path_ = testdata / "mul_2_large.onnx",
              .external_data_path_ = {},
              .inputs_ = {},   // No built-in I/O data today
              .outputs_ = {},  // No built-in I/O data today
              .has_dynamic_shapes_ = false}},

            // gemm_custom_op: GEMM operator surrounded by custom identity op.
            // Model will always partition to MLAS
            {"gemm_custom_op",
             {.name_ = "gemm_custom_op",
              .model_path_ = testdata / "gemm_custom_op.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"input", {testdata / "gemm_custom_op.in"}}},
              .outputs_ =
                  {{"output",
                    {testdata / "gemm_custom_op.out",
                     element_type::float16,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false}},

            // gemm_custom_op_external: GEMM operator surrounded by custom identity op.
            // Model will always partition to MLAS
            {"gemm_custom_op_external",
             {.name_ = "gemm_custom_op_external",
              .model_path_ = testdata / "gemm_custom_op_external.onnx",
              .external_data_path_ = testdata / "gemm_custom_op_external.bin",
              .inputs_ = {{"input", {testdata / "gemm_custom_op.in"}}},
              .outputs_ =
                  {{"output",
                    {testdata / "gemm_custom_op.out",
                     element_type::float16,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false,
              .cpu_fallback_required_ = true}},

            // gemm_external: GEMM with external initializers (fp16 weights)
            {"gemm_external",
             {.name_ = "gemm_external",
              .model_path_ = testdata / "gemm_external.onnx",
              .external_data_path_ = testdata / "gemm_external.bin",
              .inputs_ = {{"input", {testdata / "gemm_external.in", element_type::float16}}},
              .outputs_ =
                  {{"output",
                    {testdata / "gemm_external.out",
                     element_type::float16,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false}},

            // gemm_external_unicode: GEMM with Unicode-named model file and Unicode-named bin file
            {"gemm_external_unicode",
             {.name_ = "gemm_external_unicode",
              .model_path_ = testdata / fs::path(u8"\u4e2d\u6587.onnx"),
              .external_data_path_ = testdata / fs::path(u8"\u4e2d\u6587.bin"),
              .inputs_ = {{"input", {testdata / "gemm_external.in", element_type::float16}}},
              .outputs_ =
                  {{"output",
                    {testdata / "gemm_external.out",
                     element_type::float16,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false}},

            // gemm_qdq_scale_zp_external: Quantized GEMM with external initializers
            {"gemm_qdq_scale_zp_external",
             {.name_ = "gemm_qdq_scale_zp_external",
              .model_path_ = testdata / "gemm_qdq_scale_zp_external.onnx",
              .external_data_path_ = testdata / "gemm_qdq_scale_zp_external.data",
              .inputs_ = {{"input", {testdata / "gemm_qdq_scale_zp_external.in", element_type::float32}}},
              .outputs_ =
                  {{"output",
                    {testdata / "gemm_qdq_scale_zp_external.out",
                     element_type::float32,
                     comparison_strategy::cosine_similarity,
                     0.95}}},
              .has_dynamic_shapes_ = false}},

            // matmulnbits_external: tiny MatMulNBits with external B/scales/zero_points (bits=2, block_size=64, UINT8 B/zp, FLOAT scales; K=512/N=16). Exercises the
            // weights-as-input transform on the frontend's MatMulNBits decomposition. Input/output are
            // defined in code (ramp input, golden from ORT CPU); only the model + weights are on disk.
            {"matmulnbits_external",
             {.name_ = "matmulnbits_external",
              .model_path_ = testdata / "matmulnbits_external.onnx",
              .external_data_path_ = testdata / "matmulnbits_external.data",
              .inputs_ = {{"input", {ramp_floats_to_bytes(512), element_type::float32}}},
              .outputs_ =
                  {{"output",
                    {floats_to_bytes({1.431097f, 1.158935f, -1.552827f, 0.317955f, -2.460433f, -2.187771f,
                                      1.413617f, -4.540843f, 0.387316f, -0.016576f, 1.334959f, 2.596107f,
                                      -0.207024f, -0.583266f, -0.283409f, 2.125920f}),
                     element_type::float32,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false}},

            // matmulnbits_unaligned: same as above but with n_blocks_per_col (9) not a multiple of the
            // zero-point pack factor, so the frontend emits a Slice on the zero-point dequant path
            // (K=544/block_size=64). Exercises the optional-Slice branch of the dequant exclusion matcher.
            {"matmulnbits_unaligned",
             {.name_ = "matmulnbits_unaligned",
              .model_path_ = testdata / "matmulnbits_unaligned.onnx",
              .external_data_path_ = testdata / "matmulnbits_unaligned.data",
              .inputs_ = {{"input", {ramp_floats_to_bytes(544), element_type::float32}}},
              .outputs_ =
                  {{"output",
                    {floats_to_bytes({-0.844751f, 0.537437f, -2.080402f, 4.456698f, -3.161492f, 5.855534f,
                                      -0.101494f, -2.165379f, 0.529275f, -0.089314f, -3.217348f, -0.527825f,
                                      -0.210679f, -0.360664f, -0.568940f, 2.270906f}),
                     element_type::float32,
                     comparison_strategy::cosine_similarity,
                     0.99}}},
              .has_dynamic_shapes_ = false}},

            // tiny_llama3: real Llama-3.2-tiny-random ONNX export (transformers 4.55.4).
            // 2 layers, 1 KV head (GQA: 2 query heads), head_dim=4, hidden=8, vocab=128256.
            // KV cache is float32. All ops are standard ONNX domain (no custom ops). opset 21.
            // Inputs/outputs are constructed per-test (dynamic shapes).
            {"tiny_llama3",
             {.name_ = "tiny_llama3",
              .model_path_ = testdata / "tiny_llama3.onnx",
              .external_data_path_ = {},
              .inputs_ = {},
              .outputs_ = {},
              .has_dynamic_shapes_ = true}},

            // tiny_llama3_numeric_prefix: same export as tiny_llama3, with every graph input and
            // output renamed to "<digits>_<name>" to mimic the WebNN naming scheme. Exercises the
            // numeric-prefix stripping in OvComputeInfoStateful.
            // Inputs/outputs are constructed per-test (dynamic shapes).
            {"tiny_llama3_numeric_prefix",
             {.name_ = "tiny_llama3_numeric_prefix",
              .model_path_ = testdata / "tiny_llama3_numeric_prefix.onnx",
              .external_data_path_ = {},
              .inputs_ = {},
              .outputs_ = {},
              .has_dynamic_shapes_ = true}},

            // tiny_llama3_epctx_ir: same model pre-compiled to OpenVINO IR and wrapped in an
            // EPContext ONNX node (embed_mode=0, external xml+bin). Exercises the native
            // ImportNativeEpContext path. Inputs/outputs are constructed per-test (dynamic shapes).
            {"tiny_llama3_epctx_ir",
             {.name_ = "tiny_llama3_epctx_ir",
              .model_path_ = testdata / "tiny_llama3_epctx_ir.onnx",
              .external_data_path_ = {},
              .inputs_ = {},
              .outputs_ = {},
              .has_dynamic_shapes_ = true}},

            // if_simple: Simple If op with both branches supported by OpenVINO
            // X=[-1,0.5,2,-0.5,1.5,-2], cond=true → Y=relu(X)+1=[1,1.5,3,1,2.5,1]
            {"if_simple",
             {.name_ = "if_simple",
              .model_path_ = testdata / "if_simple.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({-1.0f, 0.5f, 2.0f, -0.5f, 1.5f, -2.0f})}},
                          {"cond", {bool_to_bytes(true)}}},
              .outputs_ = {{"Y", {floats_to_bytes({1.0f, 1.5f, 3.0f, 1.0f, 2.5f, 1.0f})}}},
              .has_dynamic_shapes_ = false}},

            // if_multi_output_forwarded: If op with multiple outputs including forwarded parent-scope value
            // X=[-1,0.5,2,-0.5,1.5,-2], cond=true → result=relu(X)+1, forwarded=X
            {"if_multi_output_forwarded",
             {.name_ = "if_multi_output_forwarded",
              .model_path_ = testdata / "if_multi_output_forwarded.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({-1.0f, 0.5f, 2.0f, -0.5f, 1.5f, -2.0f})}},
                          {"cond", {bool_to_bytes(true)}}},
              .outputs_ = {{"result", {floats_to_bytes({1.0f, 1.5f, 3.0f, 1.0f, 2.5f, 1.0f})}},
                           {"forwarded", {floats_to_bytes({-1.0f, 0.5f, 2.0f, -0.5f, 1.5f, -2.0f})}}},
              .has_dynamic_shapes_ = false}},

            // if_custom_op_branch: If op with unsupported custom op in else_branch
            // X=[0.1,0.5,0.9,0.3,0.7,0.2], cond=true → Y=relu(X)+1=[1.1,1.5,1.9,1.3,1.7,1.2]
            {"if_custom_op_branch",
             {.name_ = "if_custom_op_branch",
              .model_path_ = testdata / "if_custom_op_branch.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f})}},
                          {"cond", {bool_to_bytes(true)}}},
              .outputs_ = {{"Y", {floats_to_bytes({1.1f, 1.5f, 1.9f, 1.3f, 1.7f, 1.2f})}}},
              .has_dynamic_shapes_ = false,
              .cpu_fallback_required_ = true}},

            // if_nested: Nested If op with inner If in outer then_branch
            // X=[-1,0.5,2,-0.5,1.5,-2], cond_outer=true, cond_inner=true → Y=relu(X)+1
            {"if_nested",
             {.name_ = "if_nested",
              .model_path_ = testdata / "if_nested.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({-1.0f, 0.5f, 2.0f, -0.5f, 1.5f, -2.0f})}},
                          {"cond_outer", {bool_to_bytes(true)}},
                          {"cond_inner", {bool_to_bytes(true)}}},
              .outputs_ = {{"Y", {floats_to_bytes({1.0f, 1.5f, 3.0f, 1.0f, 2.5f, 1.0f})}}},
              .has_dynamic_shapes_ = false}},

            // if_multi_output: If op with multiple outputs and forwarded input
            // X=[-1,0.5,2,-0.5,1.5,-2], W=[1,2,0.5,1.5,0.5,3], cond=true
            // out_a=relu(X)+W, out_b=relu(X)*W, residual=tanh(X)
            {"if_multi_output",
             {.name_ = "if_multi_output",
              .model_path_ = testdata / "if_multi_output.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({-1.0f, 0.5f, 2.0f, -0.5f, 1.5f, -2.0f})}},
                          {"W", {floats_to_bytes({1.0f, 2.0f, 0.5f, 1.5f, 0.5f, 3.0f})}},
                          {"cond", {bool_to_bytes(true)}}},
              .outputs_ = {{"out_a", {floats_to_bytes({1.0f, 2.5f, 2.5f, 1.5f, 2.0f, 3.0f})}},
                           {"out_b", {floats_to_bytes({0.0f, 1.0f, 1.0f, 0.0f, 0.75f, 0.0f})}},
                           {"residual", {floats_to_bytes({-0.76159418f, 0.46211720f, 0.96402758f, -0.46211720f, 0.90514821f, -0.96402758f})}}},
              .has_dynamic_shapes_ = false}},

            // if_nested_custom_op: Nested If with unsupported custom op in inner else_branch
            // X=[0.1,0.5,0.9,0.3,0.7,0.2], cond_outer=true, cond_inner=true → Y=relu(X)+1
            {"if_nested_custom_op",
             {.name_ = "if_nested_custom_op",
              .model_path_ = testdata / "if_nested_custom_op.onnx",
              .external_data_path_ = {},
              .inputs_ = {{"X", {floats_to_bytes({0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f})}},
                          {"cond_outer", {bool_to_bytes(true)}},
                          {"cond_inner", {bool_to_bytes(true)}}},
              .outputs_ = {{"Y", {floats_to_bytes({1.1f, 1.5f, 1.9f, 1.3f, 1.7f, 1.2f})}}},
              .has_dynamic_shapes_ = false,
              .cpu_fallback_required_ = true}}};
  };
};

}  // namespace ovep_tests
