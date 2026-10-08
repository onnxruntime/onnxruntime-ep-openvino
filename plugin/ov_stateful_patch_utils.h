// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/make_stateful.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"

namespace onnxruntime {
namespace openvino_ep {

void LogBasicModelInfo(const std::shared_ptr<const ov::Model>& model);

bool ModelHasInputOutputNames(std::shared_ptr<ov::Model> model, const std::string& name_to_match);

void FuseCacheReorder(std::shared_ptr<ov::Model> ov_model,
                      std::vector<std::string>& not_kv_inputs,
                      const std::vector<std::string>& key_value_input_names,
                      int gather_dim,
                      const bool should_add_kvcache_reorder = false);

void MakeStateful(std::shared_ptr<ov::Model>& ov_model,
                  const std::map<std::string, std::string>& input_output_map);

void PatchStatefulDecoder(std::shared_ptr<ov::Model> model, const bool should_add_kvcache_reorder = false);

bool HasOpWithType(const std::shared_ptr<const ov::Model>& function, const std::string& type_name);

std::tuple<std::shared_ptr<ov::Node>, int64_t> FindLLMMatmul(const std::shared_ptr<ov::Model>& model);

void ApplySliceBeforeMatmulTransformation(std::shared_ptr<ov::Model> model);

void UpdateConfig(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair);

std::optional<ov::Any> PopOption(ov::AnyMap& config, const std::string& option_name);

void RenameKey(ov::AnyMap& config, const std::string& old_key, const std::string& new_key);

struct KVAxesPosition {
  size_t batch;
  size_t seq_len;
};

KVAxesPosition GetKVAxesPos(std::shared_ptr<const ov::Model> model);

struct KVDesc {
  uint32_t max_prompt_len;
  uint32_t min_response_len;
};

void UpdateNPUConfig(ov::AnyMap& config, const KVAxesPosition& kv_pos, const KVDesc& kv_desc);

std::optional<uint32_t> PopIntAndCast(ov::AnyMap& config, const std::string& key);

bool IsStateful(const std::shared_ptr<ov::Model>& model);

bool HasTensor(const ov::CompiledModel& compiled_model, const std::string& tensor_name);
;

}  // namespace openvino_ep
}  // namespace onnxruntime
