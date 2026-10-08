// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime {
///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct session_wrapper : Ort::Session {
  using session_options_t = typename fixture_t::session_options_wrapper_t;
  using io_binding_t = typename fixture_t::io_binding_wrapper_t;
  using custom_ops_t = typename fixture_t::custom_ops_wrapper_t;

  const size_t input_count_ = 0;
  const size_t output_count_ = 0;
  std::shared_ptr<custom_ops_t> custom_ops_{};  // keep custom ops alive for the life of the session

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(session_wrapper);

  ///////////////////////////////////////////////////////////////////////
  session_wrapper(const fs::path& model_path, const session_options_t& options)
      : Ort::Session(env_, model_path.native().c_str(), options), input_count_(GetInputCount()), output_count_(GetOutputCount()), custom_ops_(options.custom_ops_) {}

  ///////////////////////////////////////////////////////////////////////
  session_wrapper(const std::vector<uint8_t>& model, const session_options_t& options)
      : Ort::Session(env_, model.data(), model.size(), options), input_count_(GetInputCount()), output_count_(GetOutputCount()), custom_ops_(options.custom_ops_) {}

  ///////////////////////////////////////////////////////////////////////
  virtual ~session_wrapper(void) {}

  ///////////////////////////////////////////////////////////////////////
  void set_ep_dynamic_options(const std::map<std::string, std::string>& options) {
    std::vector<const char*> keys;
    std::vector<const char*> values;

    std::transform(
        options.begin(),
        options.end(),
        std::back_inserter(keys),
        [](const auto& pair) { return pair.first.c_str(); });

    std::transform(
        options.begin(),
        options.end(),
        std::back_inserter(values),
        [](const auto& pair) { return pair.second.c_str(); });

    const char* const* const_keys_data = keys.data();
    const char* const* const_values_data = values.data();

    SetEpDynamicOptions(const_keys_data, const_values_data, options.size());
  }

  ///////////////////////////////////////////////////////////////////////
  auto is_input(size_t index) { return index < input_count_; }

  ///////////////////////////////////////////////////////////////////////
  std::string get_tensor_name(size_t index, bool is_input) const {
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = is_input ? GetInputNameAllocated(index, allocator) : GetOutputNameAllocated(index, allocator);
    return std::string(name.get());
  }

  ///////////////////////////////////////////////////////////////////////
  auto get_type_info(size_t index, bool is_input) const {
    return is_input ? GetInputTypeInfo(index) : GetOutputTypeInfo(index);
  }

  ///////////////////////////////////////////////////////////////////////
  size_t get_element_size_in_bytes(ONNXTensorElementDataType type) const {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return 1;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
      default:
        return 0;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void run(
      const std::vector<const char*>& input_names,
      const std::vector<Ort::Value>& input_tensors,
      const std::vector<const char*>& output_names,
      std::vector<Ort::Value>& output_tensors) {
    Ort::RunOptions run_options{};

    OVEP_TESTS_LOG_DEBUG << "OrtSession::Run( session: " << this << " ): {";

    Run(run_options,
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_tensors.data(),
        output_names.size());

    OVEP_TESTS_LOG_DEBUG << "OrtSession::Run: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  void run(const io_binding_t& io_binding) {
    Ort::RunOptions run_options{};

    OVEP_TESTS_LOG_DEBUG << "OrtSession::Run( session: " << this << ", io_binding: " << io_binding << " ): {";

    Run(run_options, io_binding);

    OVEP_TESTS_LOG_DEBUG << "OrtSession::Run: ...OK }";
  }
};

//////////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct session_factory {
  using session_options_t = typename fixture_t::session_options_wrapper_t;
  using session_t = typename fixture_t::session_wrapper_t;
  using custom_ops_t = typename fixture_t::custom_ops_wrapper_t;

  ///////////////////////////////////////////////////////////////////////
  static std::unique_ptr<session_t> create(const network_t& network, const session_options_t& session_options) {
    if (!network.name_.empty()) {
      return std::make_unique<session_t>(network.model_path(), session_options);
    } else if (!network.buffer_.empty()) {
      return std::make_unique<session_t>(network.buffer_, session_options);
    }

    OVEP_TESTS_THROW("network name or buffer required to create a session");
  }
};

}  // namespace onnxruntime
}  // namespace ovep_tests
