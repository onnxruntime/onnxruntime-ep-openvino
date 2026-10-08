// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <format>
#include <algorithm>
#include <limits>
#include <regex>

#include "ov_compute_stateful.h"
#include "ov_ep_context.h"
#include "ov_ep_profiler.h"
#include "ov_stateful_patch_utils.h"
#include "ov_version.h"
#include "ov_plugin_utils.h"
#include "ov_utils.h"

using namespace onnxruntime::openvino_ep;

#define STATEFUL_LOG_LEVEL ORT_LOGGING_LEVEL_INFO

namespace onnxruntime {
namespace openvino_ep_plugin {

OvComputeInfoStateful::OvComputeInfoStateful(ov::Core& ov_core, const Ort::Logger& logger) : OvComputeInfo(ov_core, logger) {}

OrtStatus* OvComputeInfoStateful::Init(OvComputeInitParams&& params, EpContextNode&& ep_context_node) {
  OVEP_TRACE(logger_, "ep_context");

  switch (ep_context_node.private_fields_.type) {
    case EpContextNode::EpContextType::Native:
      OVEP_RETURN_IF_ERROR(ImportNativeEpContext(params, std::move(ep_context_node)));
      break;
    case EpContextNode::EpContextType::OV_IR:
      if (ep_context_node.embed_mode != 0) {
        // To support this we would need to save the weights location somewhere, or have a schema for embedding the weights along with the IR.
        return Ort::Status("Epctx with OVIR must use embed_mode == 0", ORT_INVALID_ARGUMENT).release();
      } else {
        std::shared_ptr<ov::Model> model;
        {
          OVEP_TRACE(logger_, "ov::Core::read_model(path=%s)", ToUtf8String(ep_context_node.GetEpContextPath()).c_str());
          model = ov_core_.read_model(ep_context_node.GetEpContextPath());
        }
        if (!params.transformations.empty()) {
          OVEP_RETURN_IF_ERROR(params.transformations.ApplyTransformations(model));
        }
        OVEP_RETURN_IF_ERROR(StatefulCompile(std::move(model), params.ov_device, params.configs));
      }
      break;
    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  return StatefulPostCompileInit(params.ov_device,
                                 params.model_info.io_mapping,
                                 std::move(params.transformations),
                                 params.shared_context);
}

OrtStatus* OvComputeInfoStateful::Init(OvComputeInitParams&& params, std::string&& model) {
  return InitCommon(std::move(params), std::move(model));
}

OrtStatus* OvComputeInfoStateful::Init(OvComputeInitParams&& params, const OrtGraph* graph) {
  return InitCommon(std::move(params), graph);
}

template <typename T>
OrtStatus* OvComputeInfoStateful::InitCommon(OvComputeInitParams&& params, T&& model) {
  OVEP_TRACE(logger_);

  const OnnxIOMapping& io_mapping = params.model_info.io_mapping;
  std::shared_ptr<ov::Model> ov_model;
  {
    OVEP_TRACE(logger_, "ReadModelWithExternalWeightSupport(path=%s)", ToUtf8String(params.model_info.model_path).c_str());
    OVEP_RETURN_IF_ERROR(ReadModelWithExternalWeightSupport(std::move(model), params, params.model_info.model_path, ov_model));
  }

  if (!params.transformations.empty()) {
    OVEP_RETURN_IF_ERROR(params.transformations.ApplyTransformations(ov_model));
  }

  OVEP_RETURN_IF_ERROR(StatefulCompile(std::move(ov_model), params.ov_device, params.configs));
  return StatefulPostCompileInit(params.ov_device,
                                 io_mapping,
                                 std::move(params.transformations),
                                 params.shared_context);
}

static inline bool IsNPUWSliceOutEnabled(const ov::CompiledModel& compiled_model) {
  auto slice_out_val = compiled_model.get_property("NPUW_SLICE_OUT");
  if (!slice_out_val.empty()) {
    if (slice_out_val.is<std::string>()) {
      return (slice_out_val.as<std::string>() == "YES");
    } else if (slice_out_val.is<bool>()) {
      return slice_out_val.as<bool>();
    }
  }

  return false;
}

// Matches "123_", "0_", "42_", etc. at start of string.
const std::regex kPrefixRegex{R"(^(\d+)_)"};

static inline bool starts_with_numeric_prefix(const std::string& name) {
  return std::regex_search(name, kPrefixRegex);
}

static inline bool all_names_have_numeric_prefix(const ov::OutputVector& outputs) {
  for (const auto& output : outputs) {
    const auto& names = output.get_tensor().get_names();

    if (names.empty()) {
      return false;
    }

    for (const auto& name : names) {
      if (!starts_with_numeric_prefix(name)) {
        return false;
      }
    }
  }

  return true;
}

static inline bool all_model_io_tensor_names_have_numeric_prefix(
    const std::shared_ptr<ov::Model>& model) {
  return all_names_have_numeric_prefix(model->inputs()) &&
         all_names_have_numeric_prefix(model->outputs());
}

static inline std::string strip_numeric_prefix(const std::string& name) {
  return std::regex_replace(name, kPrefixRegex, "");
}

static inline void strip_prefix_from_names(
    const ov::OutputVector& outputs) {
  for (const auto& output : outputs) {
    auto& tensor = output.get_tensor();

    const auto& old_names = tensor.get_names();
    std::unordered_set<std::string> new_names;

    for (const auto& name : old_names) {
      new_names.insert(strip_numeric_prefix(name));
    }

    tensor.set_names(new_names);
  }
}

static inline void strip_numeric_prefix_from_kv_io_tensor_names(
    const std::shared_ptr<ov::Model>& model) {
  strip_prefix_from_names(model->inputs());
  strip_prefix_from_names(model->outputs());
}

static inline bool all_onnx_io_names_have_numeric_prefix(const OnnxIOMapping& io_mapping) {
  for (const auto& name : io_mapping.input_names) {
    if (!starts_with_numeric_prefix(name)) {
      return false;
    }
  }
  for (const auto& name : io_mapping.output_names) {
    if (!starts_with_numeric_prefix(name)) {
      return false;
    }
  }
  return true;
}

OrtStatus* OvComputeInfoStateful::StatefulPostCompileInit(const OvDevice& ov_device,
                                                          const OnnxIOMapping& io_mapping,
                                                          TransformationPipeline transformations,
                                                          const std::shared_ptr<SharedContext>& shared_context) {
  auto tensor_has_type = [&](const std::string& tensor_name, const ov::element::Type& type) -> bool {
    const auto& input = compiled_model_.input(tensor_name);
    return input.get_element_type() == type;
  };

  if (HasTensor(compiled_model_, "beam_idx") && !tensor_has_type("beam_idx", ov::element::i32)) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Compiled Model must have 'beam_idx' tensor of element type i32");
  }

  has_position_ids_ = HasTensor(compiled_model_, "position_ids");
  if (has_position_ids_ && !tensor_has_type("position_ids", ov::element::i64)) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "'position_ids' tensor must be of type i64");
  }

  kvcache_reorder_enabled_ = false;
  const bool has_src_idx = HasTensor(compiled_model_, "src_idx");
  const bool has_dst_idx = HasTensor(compiled_model_, "dst_idx");
  if (has_src_idx != has_dst_idx) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                "Both 'src_idx' and 'dst_idx' tensors must be present to enable KV cache reorder support");
  }
  if (has_src_idx && has_dst_idx) {
    if (!tensor_has_type("src_idx", ov::element::i32) || !tensor_has_type("dst_idx", ov::element::i32)) {
      return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                  "'src_idx' and 'dst_idx' tensors must be of type i32 when KV cache reorder is enabled");
    }

    kvcache_reorder_enabled_ = true;
  }

  _infer_request = std::make_unique<WrappedInferRequest>(std::move(compiled_model_.create_infer_request()));
  if (!transformations.empty()) {
    transformations.InitializeInferRequest(_infer_request->ov());
  }

  // Check WebNN use case, in which all ONNX input / output names have numeric prefixes.
  if (all_onnx_io_names_have_numeric_prefix(io_mapping)) {
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "All ONNX IO names have numeric prefix");
    // In this case, we will strip the numeric prefixes from the ONNX IO names when creating the bindings,
    // since the underlying OV model has also stripped these.
    auto stripped_io_mapping = io_mapping;
    for (auto& name : stripped_io_mapping.input_names) {
      name = strip_numeric_prefix(name);
    }
    for (auto& name : stripped_io_mapping.output_names) {
      name = strip_numeric_prefix(name);
    }
    onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(
        compiled_model_, stripped_io_mapping, SessionContext{true}, shared_context);
  } else {
    onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(
        compiled_model_, io_mapping, SessionContext{true}, shared_context);
  }

  bool gpu_or_npu = (ov_device.Contains("NPU") || ov_device.Contains("GPU"));

  // If GPU or NPU are being used, and if this model takes in either "input_ids" or "inputs_embeds",
  // enable use of chat history caching. It's enabled for GPU right now as get_state API is supported only
  // when KV-cache compression is disabled, and it's enabled by default.
  prefill_use_full_chat_history_ = false;
  if (gpu_or_npu) {
    const bool has_input_ids = HasTensor(compiled_model_, "input_ids");
    const bool has_inputs_embeds = HasTensor(compiled_model_, "inputs_embeds");

    OVEP_RETURN_IF(has_input_ids && has_inputs_embeds, ort_api,
                   "Compiled Model must not expose both 'input_ids' and 'inputs_embeds'; full chat "
                   "history caching supports only one primary sequence tensor per compiled model.");

    if (has_input_ids || has_inputs_embeds) {
      prefill_use_full_chat_history_ = true;
      primary_seq_tensor_is_embeds_ = has_inputs_embeds;
      primary_seq_tensor_name_ = has_inputs_embeds ? "inputs_embeds" : "input_ids";

      if (primary_seq_tensor_is_embeds_) {
        if (!tensor_has_type(primary_seq_tensor_name_, ov::element::f32)) {
          return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                      "Compiled Model must have 'inputs_embeds' tensor of element type f32 when prefill uses full chat history.");
        }
        const auto& embeds_shape = compiled_model_.input(primary_seq_tensor_name_).get_partial_shape();
        OVEP_RETURN_IF(!(embeds_shape.rank().is_static() && embeds_shape.rank().get_length() == 3),
                       ort_api,
                       "'inputs_embeds' tensor must have static rank 3 ([batch, seq_len, hidden_size]).");
        OVEP_RETURN_IF(!embeds_shape[2].is_static(),
                       ort_api,
                       "'inputs_embeds' tensor's hidden-size dimension (axis 2) must be static.");
        embeds_hidden_size_ = gsl::narrow<size_t>(embeds_shape[2].get_length());
        OVEP_RETURN_IF(embeds_hidden_size_ == 0,
                       ort_api,
                       "'inputs_embeds' tensor's hidden-size dimension (axis 2) must be positive.");
      } else {
        if (!tensor_has_type(primary_seq_tensor_name_, ov::element::i64)) {
          return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                      "Compiled Model must have 'input_ids' tensor of element type i64 when prefill uses full chat history.");
        }
      }
    }
  }

  // NPUW returns logits from its internal requests rather than writing to the externally bound output,
  // even when NPUW_SLICE_OUT is enabled. Keep copying those logits into the ORT output.
  preallocated_outputs_supported_ = !ov_device.Contains("NPU");
  if (ov_device.Contains("NPU")) {
    // If NPUW_SLICE_OUT is enabled, it means that it's not required to slice within OVEP.
    // Otherwise, if NPUW_SLICE_OUT is NOT enabled, then we need to perform some explicit logit
    // slicing in OVEP.
    _npu_logits_slice_required = !IsNPUWSliceOutEnabled(compiled_model_);
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::PostInfer() {
  OVEP_RETURN_IF_ERROR(CleanReorderKVCacheStatus());
  return nullptr;
}

static void LogBasicModelInfo(const std::shared_ptr<const ov::Model>& model, const Ort::Logger& logger) {
  if (STATEFUL_LOG_LEVEL < logger.GetLoggingSeverityLevel()) {
    return;
  }
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "Model Name: %s", model->get_friendly_name().c_str());
  // Log detailed information about model inputs and outputs
  auto inputs = model->inputs();
  auto outputs = model->outputs();
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\tInputs: ");
  for (const ov::Output<const ov::Node>& input : inputs) {
    const std::string name = input.get_any_name();
    const ov::element::Type type = input.get_element_type();
    const ov::PartialShape shape = input.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(input);
    ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\t\t%s, %s, %s, %s", name.c_str(), type.to_string().c_str(), shape.to_string().c_str(), layout.to_string().c_str());
  }
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\tOutputs: ");
  for (const ov::Output<const ov::Node>& output : outputs) {
    const std::string name = output.get_any_name();
    const ov::element::Type type = output.get_element_type();
    const ov::PartialShape shape = output.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(output);
    ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\t\t%s, %s, %s, %s", name.c_str(), type.to_string().c_str(), shape.to_string().c_str(), layout.to_string().c_str());
  }
}

static inline bool HasGQA(const std::shared_ptr<ov::Model>& model) {
  for (const auto& op : model->get_ops()) {
    const std::string type_name = op->get_type_name();

    // ONNX's GroupQueryAttention op is mapped to ov::op::internal::GroupQueryAttention by the frontend
    if (type_name.find("GroupQueryAttention") != std::string::npos) {
      return true;
    }
  }
  return false;
}

OrtStatus* OvComputeInfoStateful::StatefulCompile(std::shared_ptr<ov::Model> model, const OvDevice& ov_device, const ov::AnyMap& device_config) {
  ov::AnyMap config = device_config;

  ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model input to Stateful Compile:");
  LogBasicModelInfo(model, logger_);

  // Check WebNN use case, in which all ONNX input / output names have numeric prefixes.
  if (all_model_io_tensor_names_have_numeric_prefix(model)) {
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "All model IO tensor names have numeric prefix");
    // If it's the case, then we strip the numeric prefix from all input / output tensors.
    strip_numeric_prefix_from_kv_io_tensor_names(model);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model after stripping numeric prefix:");
    LogBasicModelInfo(model, logger_);
  }

  bool is_stateful = IsStateful(model);
  ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model IsStateful(): %s", is_stateful ? "True" : "False");
  const bool should_add_kvcache_reorder = (!is_stateful && ov_device.Contains("GPU") && HasGQA(model));

  if (!is_stateful) {
    PatchStatefulDecoder(model, should_add_kvcache_reorder);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model after stateful transformation:");
    LogBasicModelInfo(model, logger_);
  }

  auto kv_pos = GetKVAxesPos(model);
  if (ov_device.Contains("NPU")) {
    KVDesc kv_desc;
    kv_desc.max_prompt_len = PopIntAndCast(config, "MAX_PROMPT_LEN").value_or(1024u);
    kv_desc.min_response_len = PopIntAndCast(config, "MIN_RESPONSE_LEN").value_or(128u);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    OVEP_ENFORCE(kv_desc.max_prompt_len != 0 && kv_desc.min_response_len != 0, "MAX_PROMPT_LEN or MIN_RESPONSE_LEN cannot be 0");

    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.batch = %zu", kv_pos.batch);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.seq_len = %zu", kv_pos.seq_len);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.max_prompt_len = %u", kv_desc.max_prompt_len);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.min_response_len = %u", kv_desc.min_response_len);
    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // It's not needed for NPU, as it is already internally applied inside NPUW compilation.
    ApplySliceBeforeMatmulTransformation(model);
  }

  {
    OVEP_TRACE(logger_, "ov::Core::compile_model(model, device=%s)", ov_device.c_str());
    compiled_model_ = ov_core_.compile_model(model, ov_device, config);
  }

  return nullptr;
}

static inline void FillTensor(ov::InferRequest infer_request, const std::string& tensor_name, const ov::element::Type& type,
                              const std::vector<size_t>& shape, int32_t fill_value) {
  ov::Tensor tensor = ov::Tensor(type, shape);
  std::fill_n(tensor.data<int32_t>(), tensor.get_size(), fill_value);
  infer_request.set_tensor(tensor_name, tensor);
}

template <typename T>
static inline void CacheTensor(ov::InferRequest infer_request, const std::string& tensor_name, std::vector<T>& cache) {
  auto tensor = infer_request.get_tensor(tensor_name);
  OVEP_ENFORCE(tensor.get_element_type() == ov::element::from<T>(),
               "CacheTensor: unexpected element type for '", tensor_name, "'");
  auto* pData = tensor.data<T>();
  const size_t old_size = cache.size();
  const size_t append_size = tensor.get_size();
  cache.resize(old_size + append_size);
  std::memcpy(cache.data() + old_size, pData, append_size * sizeof(T));
}

static inline void SetTensorFromCache(ov::InferRequest infer_request, const std::string& tensor_name,
                                      const std::vector<int64_t>& cache_data) {
  auto tensor = infer_request.get_tensor(tensor_name);
  OVEP_ENFORCE(tensor.get_element_type() == ov::element::i64,
               "SetTensorFromCache: expected i64 tensor for '", tensor_name, "'");
  auto new_shape = tensor.get_shape();
  OVEP_ENFORCE(new_shape.size() == 2,
               "SetTensorFromCache: expected rank 2 tensor for '", tensor_name, "', got rank ", new_shape.size());
  new_shape[1] = cache_data.size();

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  OVEP_ENFORCE(new_tensor.get_size() == cache_data.size(),
               "SetTensorFromCache: tensor element count does not match cached data size for '", tensor_name, "'");
  auto* pNewData = new_tensor.data<int64_t>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(int64_t));

  infer_request.set_tensor(tensor_name, new_tensor);
}

// Embeddings-specific counterpart for "inputs_embeds" (fp32, rank 3, [1, seq_len, hidden_size])
// used instead of SetTensorFromCache when this compiled model's input is
// "inputs_embeds" rather than "input_ids"
static inline void SetEmbeddingsTensorFromCache(ov::InferRequest infer_request, const std::string& tensor_name,
                                                const std::vector<float>& cache_data, size_t hidden_size) {
  auto tensor = infer_request.get_tensor(tensor_name);
  OVEP_ENFORCE(tensor.get_element_type() == ov::element::f32,
               "SetEmbeddingsTensorFromCache: expected f32 tensor for '", tensor_name, "'");
  auto new_shape = tensor.get_shape();
  OVEP_ENFORCE(new_shape.size() == 3,
               "SetEmbeddingsTensorFromCache: expected rank 3 tensor for '", tensor_name, "', got rank ", new_shape.size());
  OVEP_ENFORCE(hidden_size > 0 && cache_data.size() % hidden_size == 0,
               "SetEmbeddingsTensorFromCache: cached data size is not a multiple of hidden_size for '", tensor_name, "'");
  new_shape[1] = cache_data.size() / hidden_size;
  new_shape[2] = hidden_size;

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  OVEP_ENFORCE(new_tensor.get_size() == cache_data.size(),
               "SetEmbeddingsTensorFromCache: tensor element count does not match cached data size for '", tensor_name, "'");
  auto* pNewData = new_tensor.data<float>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(float));

  infer_request.set_tensor(tensor_name, new_tensor);
}

OrtStatus* OvComputeInfoStateful::PrepareKVCacheReorderInputs(ov::InferRequest& infer_request) {
  if (!kvcache_reorder_enabled_) {
    return nullptr;
  }

  auto dst_tensor = infer_request.get_tensor("dst_idx");
  auto dst_shape = dst_tensor.get_shape();
  OVEP_RETURN_IF(dst_shape.size() != 4,
                 ort_api,
                 "'dst_idx' tensor must be rank 4 when KV cache reorder is enabled");
  const size_t kv_num_heads = dst_shape[1];
  const size_t kv_head_size = dst_shape[3];

  if (!kv_src_indices_.empty()) {
    OVEP_ENFORCE(kv_src_indices_.size() == kv_dst_indices_.size(),
                 "KV cache reorder source and destination index counts must match");

    ov::Tensor src_idx_tensor(ov::element::i32, {kv_src_indices_.size()});
    auto* src_ptr = src_idx_tensor.data<int32_t>();
    std::copy(kv_src_indices_.begin(), kv_src_indices_.end(), src_ptr);
    infer_request.set_tensor("src_idx", src_idx_tensor);

    ov::Shape dst_idx_shape{1, kv_num_heads, kv_dst_indices_.size(), kv_head_size};
    ov::Tensor dst_idx_tensor(ov::element::i32, dst_idx_shape);
    auto* dst_ptr = dst_idx_tensor.data<int32_t>();
    const size_t stride = kv_head_size;
    for (size_t head = 0; head < kv_num_heads; ++head) {
      for (size_t i = 0; i < kv_dst_indices_.size(); ++i) {
        std::fill_n(dst_ptr + (head * kv_dst_indices_.size() + i) * stride, stride, kv_dst_indices_[i]);
      }
    }
    infer_request.set_tensor("dst_idx", dst_idx_tensor);
  } else {
    FillTensor(infer_request, "src_idx", ov::element::i32, {0}, 0);
    FillTensor(infer_request, "dst_idx", ov::element::i32, {1, kv_num_heads, 0, kv_head_size}, 0);
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::PreInfer() {
  auto& infer_request = _infer_request->ov();

  OVEP_RETURN_IF_ERROR(PrepareKVCacheReorderInputs(infer_request));

  if (HasTensor(compiled_model_, "beam_idx")) {
    // Workaround: Setting the value here as it cannot be set at the ORT GenAI layer currently.
    // TODO(ankit): Address this issue and implement the fix at the appropriate layer.
    FillTensor(infer_request, "beam_idx", ov::element::i32, {1}, 0);
  }

  // If 'prefill use full chat history' mode is enabled, we need to cache either "input_ids" or "inputs_embeds" and position_ids.
  if (prefill_use_full_chat_history_) {
    auto primary_tensor = infer_request.get_tensor(primary_seq_tensor_name_);
    const auto primary_shape = primary_tensor.get_shape();
    const size_t expected_rank = primary_seq_tensor_is_embeds_ ? 3 : 2;
    OVEP_RETURN_IF(primary_shape.size() != expected_rank,
                   ort_api,
                   std::format("'{}' tensor must have rank {}, got rank {}",
                               primary_seq_tensor_name_, expected_rank, primary_shape.size())
                       .c_str());

    size_t cached_token_count = 0;
    if (primary_seq_tensor_is_embeds_) {
      CacheTensor(infer_request, primary_seq_tensor_name_, cached_inputs_embeds_);
      cached_token_count = cached_inputs_embeds_.size() / embeds_hidden_size_;
    } else {
      CacheTensor(infer_request, primary_seq_tensor_name_, cached_input_ids_);
      cached_token_count = cached_input_ids_.size();
    }

    // "position_ids" (GQA with Rotary Embeddings doesnt have position_ids) - check if exists
    if (has_position_ids_) {
      CacheTensor(infer_request, "position_ids", cached_position_ids_);
    }

    // If we're about to run the prefill model
    const size_t current_token_count = primary_shape[1];
    if (current_token_count > 1) {
      // Check if the size of the current tensor does not match the accumulated cache.
      // This indicates that we are running a subsequent prompt (not the initial prefill).
      if (current_token_count != cached_token_count) {
        // Clear the internal KVCache state. For NPU device, this operation is a no-op.
        infer_request.reset_state();

        // Set tensors using cached values
        if (primary_seq_tensor_is_embeds_) {
          SetEmbeddingsTensorFromCache(infer_request, primary_seq_tensor_name_, cached_inputs_embeds_, embeds_hidden_size_);
        } else {
          SetTensorFromCache(infer_request, primary_seq_tensor_name_, cached_input_ids_);
        }

        // Only override position_ids if it exists and we have cached values
        if (has_position_ids_ && !cached_position_ids_.empty()) {
          SetTensorFromCache(infer_request, "position_ids", cached_position_ids_);
        }
      }
    }
  }

  return nullptr;
}

// slice out the last set of logits (:, -1, :), which are required for sampling.
OrtStatus* OvComputeInfoStateful::SliceLogits(ov::Tensor& logits) {
  OVEP_RETURN_IF(logits.get_shape().size() != 3,
                 ort_api,
                 std::format("Expected logits to have shape of rank 3, but it is {}", logits.get_shape().size()).c_str());

  // Only perform slicing when the second dimension (sequence length) is greater than 1.
  // If sequence length == 1, the last set of logits is already present and no slicing is needed.
  if (logits.get_shape()[1] > 1) {
    const ov::Coordinate begin = {0, logits.get_shape()[1] - 1, 0};
    const ov::Coordinate end = {logits.get_shape()[0], logits.get_shape()[1], logits.get_shape()[2]};
    auto sliced_logits = ov::Tensor(logits, begin, end);
    if (sliced_logits.is_continuous()) {
      logits = sliced_logits;
    } else {
      auto continuous_sliced_logits = ov::Tensor(sliced_logits.get_element_type(), sliced_logits.get_shape());
      sliced_logits.copy_to(continuous_sliced_logits);
      logits = continuous_sliced_logits;
    }
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::Compute(void* /*compute_state*/,
                                          OrtKernelContext* kernel_context) {
  Ort::KernelContext context(kernel_context);

  if (!onnx_to_ov_bindings_->has_dynamic_io_) {
    return Ort::Status("Stateful Compute requires dynamic IO", ORT_RUNTIME_EXCEPTION).release();
  }

  std::vector<UserOutputBinding> user_output_bindings;
  bool user_outputs_ready = false;
  if (preallocated_outputs_supported_) {
    OVEP_RETURN_IF_ERROR(TryInitPreallocatedOutputBindings(kernel_context, user_output_bindings, user_outputs_ready));
  }

  // NPU logits slicing produces a new view (and may require a copy), so it cannot be
  // satisfied by binding the caller's final output tensor directly.
  const bool use_user_outputs = user_outputs_ready && !_npu_logits_slice_required;
  if (stateful_output_mode_ == StatefulOutputMode::kUserBound && !use_user_outputs) {
    return OVEP_MAKE_STATUS(
        "Stateful causal-LLM execution cannot switch from user-provided output buffers to dynamically "
        "allocated outputs. Provide all required output buffers for every subsequent inference.",
        ORT_EP_FAIL);
  }

  // Dynamic shape inference

  // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
  // tensors instead of binding them to the infer request directly.

  // Bind inputs
  for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
    // Set the input shape based on the input tensor from ort
    auto tensor = context.GetInput(input_info.onnx_index);
    auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto input_shape = ParameterShape(ort_shape);

    _infer_request->SetTensor(input_info.name,
                              input_info.type,
                              input_shape,
                              const_cast<void*>(tensor.GetTensorRawData()));
  }

  // pre-inference
  OVEP_RETURN_IF_ERROR(PreInfer());

  if (use_user_outputs) {
    // This is the first point at which an externally owned output is actually bound. The
    // transition is intentionally sticky because replacing the request would lose its KV state.
    stateful_output_mode_ = StatefulOutputMode::kUserBound;
    for (const auto& binding : user_output_bindings) {
      _infer_request->SetTensor(binding.output_info->name,
                                binding.output_info->type,
                                binding.tensor.get_shape(),
                                const_cast<void*>(binding.tensor.data()));
    }
  }

  // Run Inference
  {
    OVEP_TRACE(logger_, "ov::InferRequest::infer() [stateful]");
    _infer_request->Infer();
  }
  OVEpProfilerImpl::CaptureInferRequestProfiling(_infer_request->ov());
  OVEP_RETURN_IF_ERROR(PostInfer());

  if (use_user_outputs) {
    for (const auto& binding : user_output_bindings) {
      const auto actual_shape =
          ParameterShape::ToOrtShape(_infer_request->ov().get_tensor(binding.output_info->name).get_shape());
      OVEP_RETURN_STATUS_IF(actual_shape != binding.ort_shape,
                            ORT_EP_FAIL,
                            std::format("Output tensor shape mismatch for {}", binding.output_info->name).c_str());
    }
    return nullptr;
  }

  // Copy outputs
  for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
    auto ov_tensor = _infer_request->ov().get_tensor(output_info.name);

    if (_npu_logits_slice_required && output_info.name == "logits") {
      OVEP_RETURN_IF_ERROR(SliceLogits(ov_tensor));
    }

    auto output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
    auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);
    OVEP_RETURN_IF(ov_tensor.get_byte_size() != ort_tensor.GetTensorSizeInBytes(),
                   ort_api,
                   std::format("Output tensor size mismatch for {}", output_info.name).c_str());

    if (ov_tensor.get_byte_size() != 0) {
      std::memcpy(ort_tensor.GetTensorMutableRawData(),
                  ov_tensor.data(),
                  ov_tensor.get_byte_size());
    }
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::KVCacheRewind(const size_t& index) {
  OVEP_TRACE(logger_, "KVCacheRewind(index=%zu)", index);

  auto& infer_request = _infer_request->ov();

  if (kvcache_reorder_enabled_) {
    if (index > 0) {
      // TODO: Trigger a KVCache eviction inference pass here to physically compact the KV cache
      // by running infer_request with the src_idx/dst_idx reorder tensors set.
      return ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                  ("KVCache eviction via reorder inference is not yet implemented which is required to rewind KVCache with index > 0 when KVCache reorder optimization is enabled. Requested index: " + std::to_string(index)).c_str());
    }
    OVEP_RETURN_IF_ERROR(CleanReorderKVCacheStatus());
  }

  if (prefill_use_full_chat_history_) {
    // Clear the internal KVCache state. For NPU device, this operation is a no-op.
    {
      OVEP_TRACE(logger_, "ov::InferRequest::reset_state()");
      infer_request.reset_state();
    }

    // Resize the cached "input_ids" or "inputs_embeds" and "position_ids" to the specified token index.
    if (primary_seq_tensor_is_embeds_) {
      OVEP_RETURN_IF(embeds_hidden_size_ != 0 && index > std::numeric_limits<size_t>::max() / embeds_hidden_size_,
                     ort_api,
                     "KVCacheRewind: 'index * embeds_hidden_size_' would overflow");
      const size_t target_elems = index * embeds_hidden_size_;
      if (cached_inputs_embeds_.size() > target_elems) {
        cached_inputs_embeds_.resize(target_elems);
      }
    } else {
      if (cached_input_ids_.size() > index) {
        cached_input_ids_.resize(index);
      }
    }

    if (cached_position_ids_.size() > index) {
      cached_position_ids_.resize(index);
    }
  } else {
    // If index == 0, we are clearing the entire KVCache, so simply reset the state.
    if (index == 0) {
      {
        OVEP_TRACE(logger_, "ov::InferRequest::reset_state()");
        infer_request.reset_state();
      }
      return nullptr;
    }
    // Retrieve KVCache states and trim them to the specified index.
    // The following logic is adapted from:
    // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
    std::vector<ov::VariableState> states;
    {
      OVEP_TRACE(logger_, "ov::InferRequest::query_state()");
      states = infer_request.query_state();
    }
    for (auto& state : states) {
      ov::Tensor old_tensor = state.get_state();
      // Tensor shape: [batch_size, num_kv_heads, seq_len, head_size]
      auto shape = old_tensor.get_shape();

      OVEP_ENFORCE(shape.size() == 4, "KVCacheRewind: expected 4D KV-cache state, got rank ", shape.size());

      if (shape[2] > index) {
        // Update the sequence length dimension to the specified index.
        shape[2] = index;

        ov::Coordinate new_shape_begin{0, 0, 0, 0};
        ov::Coordinate new_shape_end{shape};

        // Create a trimmed tensor with the updated shape.
        auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

        // Copy the trimmed tensor into a new tensor and update the state.
        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed_tensor.copy_to(new_tensor);

        state.set_state(new_tensor);
      }
    }
  }
  return nullptr;
}

OrtStatus* OvComputeInfoStateful::CleanReorderKVCacheStatus() {
  if (!kvcache_reorder_enabled_) {
    return nullptr;
  }

  kv_src_indices_.clear();
  kv_dst_indices_.clear();
  return nullptr;
}

OrtStatus* OvComputeInfoStateful::SetReorderKVCacheStatus(const std::vector<int32_t>& src_indices,
                                                          const std::vector<int32_t>& dst_indices) {
  if (!kvcache_reorder_enabled_) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "KV cache reorder is not enabled for this model/device");
  }

  if (src_indices.size() != dst_indices.size()) {
    auto msg = std::format("KV cache reorder requires matching src/dst counts but received {} and {}",
                           src_indices.size(), dst_indices.size());
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, msg.c_str());
  }

  kv_src_indices_ = src_indices;
  kv_dst_indices_ = dst_indices;
  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
