// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ov_compute.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

enum class StatefulOutputMode {
  kDynamicAllocation,
  kUserBound,
};

struct OvComputeInfoStateful : OvComputeInfo {
  OvComputeInfoStateful(ov::Core& ov_core, const Ort::Logger& logger);

  OrtStatus* Init(OvComputeInitParams&& params, EpContextNode&& ep_context_node) override;
  OrtStatus* Init(OvComputeInitParams&& params, std::string&& model) override;
  OrtStatus* Init(OvComputeInitParams&& params, const OrtGraph* graph) override;

  OrtStatus* Compute(void* compute_state,
                     OrtKernelContext* kernel_context) override;

  OrtStatus* KVCacheRewind(const size_t& index) override;
  OrtStatus* SetReorderKVCacheStatus(const std::vector<int32_t>& src_indices,
                                     const std::vector<int32_t>& dst_indices) override;

 protected:
  template <typename T>
  OrtStatus* InitCommon(OvComputeInitParams&& params, T&& model);

 private:
  OrtStatus* StatefulCompile(std::shared_ptr<ov::Model> model, const OvDevice& ov_device, const ov::AnyMap& device_config);
  OrtStatus* PreInfer();
  OrtStatus* PostInfer();
  OrtStatus* StatefulPostCompileInit(const OvDevice& ov_device,
                                     const OnnxIOMapping& io_mapping,
                                     TransformationPipeline transformations,
                                     const std::shared_ptr<SharedContext>& shared_context);
  OrtStatus* PrepareKVCacheReorderInputs(ov::InferRequest& infer_request);
  OrtStatus* CleanReorderKVCacheStatus();

  std::unique_ptr<WrappedInferRequest> _infer_request;

  // If prefill_use_full_chat_history_ is true, cache either "input_ids" or "inputs_embeds"
  // &"position_ids" tensors, and ensure that full chat history is passed for each prefill call.
  bool prefill_use_full_chat_history_ = false;
  bool primary_seq_tensor_is_embeds_ = false;
  std::string primary_seq_tensor_name_;
  size_t embeds_hidden_size_ = 0;
  std::vector<int64_t> cached_input_ids_;
  std::vector<float> cached_inputs_embeds_;
  std::vector<int64_t> cached_position_ids_;
  bool has_position_ids_ = false;
  bool kvcache_reorder_enabled_ = false;
  std::vector<int32_t> kv_src_indices_;
  std::vector<int32_t> kv_dst_indices_;

  OrtStatus* SliceLogits(ov::Tensor& ov_tensor);
  bool _npu_logits_slice_required = false;
  bool preallocated_outputs_supported_ = true;
  StatefulOutputMode stateful_output_mode_ = StatefulOutputMode::kDynamicAllocation;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
