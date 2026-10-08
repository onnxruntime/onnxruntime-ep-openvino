// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <limits>

#include "ov_utils.h"
#include "ov_stateful_patch_utils.h"
#include "openvino/runtime/compiled_model.hpp"

namespace onnxruntime {
namespace openvino_ep {

void LogBasicModelInfo(const std::shared_ptr<const ov::Model>& model) {
  std::cout << "Model Name: " << model->get_friendly_name() << std::endl;

  // Log detailed information about model inputs and outputs
  auto inputs = model->inputs();
  auto outputs = model->outputs();

  std::cout << "\tInputs: " << std::endl;
  for (const ov::Output<const ov::Node>& input : inputs) {
    const std::string name = input.get_any_name();
    const ov::element::Type type = input.get_element_type();
    const ov::PartialShape shape = input.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(input);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  std::cout << "\tOutputs: " << std::endl;
  for (const ov::Output<const ov::Node>& output : outputs) {
    const std::string name = output.get_any_name();
    const ov::element::Type type = output.get_element_type();
    const ov::PartialShape shape = output.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(output);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  return;
}

bool ModelHasInputOutputNames(std::shared_ptr<ov::Model> model, const std::string& name_to_match) {
  for (const ov::Output<ov::Node>& input : model->inputs()) {
    auto& names = input.get_names();

    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  for (const ov::Output<ov::Node>& output : model->outputs()) {
    auto& names = output.get_names();
    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  return false;
}

std::string GetInputOutputName(std::shared_ptr<ov::Model> ov_model,
                               const std::vector<std::string>& candidate_names) {
  for (const auto& name : candidate_names) {
    if (ModelHasInputOutputNames(ov_model, name)) {
      return name;
    }
  }
  // Return the first candidate as default if none are found
  return candidate_names.empty() ? "" : candidate_names[0];
}

void FuseCacheReorder(std::shared_ptr<ov::Model> ov_model,
                      std::vector<std::string>& not_kv_inputs,
                      const std::vector<std::string>& key_value_input_names,
                      int gather_dim,
                      const bool should_add_kvcache_reorder) {
  OVEP_ENFORCE(!ModelHasInputOutputNames(ov_model, "beam_idx"), "Model already has fused cache");

  OVEP_ENFORCE(!ModelHasInputOutputNames(ov_model, "src_idx") && !ModelHasInputOutputNames(ov_model, "dst_idx"),
               "Model already has reorder feature for KV cache");

  // Define input name candidates in priority order
  const std::vector<std::string> input_name_candidates = {
      "inputs_embeds",                       // Default fallback
      "input_ids",                           // Most common
      "input_hidden_states",                 // Alternative
      "/model/embed_tokens/Gather_output_0"  // Specific model type
  };

  std::string main_input_name = GetInputOutputName(ov_model, input_name_candidates);
  auto input_batch = ov_model->input(main_input_name).get_partial_shape()[0];
  auto update_shape = ov_model->input(key_value_input_names[0]).get_partial_shape();

  std::shared_ptr<ov::opset13::Parameter> beam_idx;
  std::shared_ptr<ov::opset13::Parameter> src_idx;
  std::shared_ptr<ov::opset13::Parameter> dst_idx;

  if (should_add_kvcache_reorder) {
    src_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, ov::PartialShape({update_shape[2]}));
    src_idx->set_friendly_name("src_idx");
    src_idx->output(0).get_tensor().add_names({"src_idx"});
    ov_model->add_parameters({src_idx});
    not_kv_inputs.push_back(src_idx->get_friendly_name());

    dst_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, update_shape);
    dst_idx->set_friendly_name("dst_idx");
    dst_idx->output(0).get_tensor().add_names({"dst_idx"});
    ov_model->add_parameters({dst_idx});
    not_kv_inputs.push_back(dst_idx->get_friendly_name());
  } else {
    beam_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, ov::PartialShape({std::move(input_batch)}));
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).get_tensor().add_names({"beam_idx"});
    ov_model->add_parameters({beam_idx});
    not_kv_inputs.push_back(beam_idx->get_friendly_name());
  }

  // Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
  for (const auto& input_name : key_value_input_names) {
    auto parameter_output_port = ov_model->input(input_name);
    auto consumers = parameter_output_port.get_target_inputs();

    std::shared_ptr<ov::Node> output_node;
    if (should_add_kvcache_reorder) {
      auto updatekv_gather_op =
          std::make_shared<ov::opset13::Gather>(parameter_output_port,
                                                src_idx,
                                                ov::opset13::Constant::create(ov::element::i64, {}, {2}));

      auto updatekv_op = std::make_shared<ov::opset12::ScatterElementsUpdate>(parameter_output_port,
                                                                              dst_idx,
                                                                              updatekv_gather_op,
                                                                              ov::opset13::Constant::create(ov::element::i64, {}, {2}));
      output_node = updatekv_op;
    } else {
      output_node = std::make_shared<ov::opset13::Gather>(parameter_output_port,
                                                          beam_idx,
                                                          ov::opset13::Constant::create(ov::element::i64, {}, {gather_dim}));
    }

    // Replace the source output for all consumers of the input tensor
    for (auto& consumer : consumers) {
      consumer.replace_source_output(output_node->output(0));
    }
  }

  // Validate the modified model
  ov_model->validate_nodes_and_infer_types();
}

// Parse tensor names into components for pattern matching
std::vector<std::string> ParseInputOutput(const std::string& name) {
  std::vector<std::string> parsed_tensor;
  std::string word;

  for (char c : name) {
    if (c == '_' || c == '.') {
      if (!word.empty()) {
        parsed_tensor.push_back(word);
        word.clear();
      }
    } else {
      word += c;
    }
  }
  if (!word.empty()) parsed_tensor.push_back(word);

  return parsed_tensor;
}

// Match KV Input tensors to the correct KV output tensors
std::map<std::string, std::string> MatchKVTensors(const std::vector<std::string>& key_value_input_names,
                                                  const std::vector<std::vector<std::string>>& parsed_key_value_input,
                                                  const std::vector<std::string>& key_value_output_names,
                                                  const std::vector<std::vector<std::string>>& parsed_key_value_output) {
  // Returns {layer_number, inner (between first word and layer), suffix (after layer)}
  auto ExtractLayerParts = [](const std::vector<std::string>& parsed_tensor)
      -> std::tuple<std::string, std::vector<std::string>, std::vector<std::string>> {
    for (size_t i = 0; i < parsed_tensor.size(); ++i) {
      const auto& t = parsed_tensor[i];
      if (!t.empty() && std::all_of(t.begin(), t.end(), ::isdigit)) {
        // inner = everything between first word (index 0) and layer (index i)
        std::vector<std::string> inner =
            i > 1 ? std::vector<std::string>(parsed_tensor.begin() + 1, parsed_tensor.begin() + i)
                  : std::vector<std::string>{};
        std::vector<std::string> suffix(parsed_tensor.begin() + i + 1, parsed_tensor.end());
        return {t, inner, suffix};
      }
    }
    return {"", {}, {}};
  };

  std::map<std::string, std::string> input_output_map;

  for (size_t i = 0; i < parsed_key_value_input.size(); ++i) {
    auto [input_layer, input_inner, input_suffix] = ExtractLayerParts(parsed_key_value_input[i]);

    bool matched = false;
    for (size_t j = 0; j < parsed_key_value_output.size(); ++j) {
      auto [output_layer, output_inner, output_suffix] = ExtractLayerParts(parsed_key_value_output[j]);

      if (input_layer != output_layer) continue;

      // If suffix exists on both sides, match on suffix (key/value distinction)
      // If no suffix on either side, fall back to inner (between first word and layer)
      bool match = (input_suffix.empty() && output_suffix.empty())
                       ? input_inner == output_inner
                       : input_suffix == output_suffix;

      if (match) {
        input_output_map[key_value_input_names[i]] = key_value_output_names[j];
        matched = true;
        break;
      }
    }

    OVEP_ENFORCE(matched, "No output match found for input ", key_value_input_names[i]);
  }
  return input_output_map;
}

void MakeStateful(std::shared_ptr<ov::Model>& ov_model,
                  const std::map<std::string, std::string>& input_output_map) {
  // Apply the transformation to make the model stateful
  ov::pass::Manager manager;
  manager.register_pass<ov::pass::MakeStateful>(input_output_map);
  manager.run_passes(ov_model);
}

// Helper function to parse output names
//
// Example: Given output names ["present_key_cross_0", "present_key_cross_1", "present_value_cross_0", "present_value_cross_1", "logits"]
//   key_value_output_names = ["present_key_cross_0", "present_key_cross_1", "present_value_cross_0", "present_value_cross_1"]
//   parsed_key_value_output = [["present", "key", "cross", "0"], ["present", "key", "cross", "1"], ["present", "value", "cross", "0"], ["present", "value", "cross", "1"]]
std::pair<std::vector<std::vector<std::string>>, std::vector<std::string>> ExtractOutputKVTensors(const std::shared_ptr<ov::Model>& model) {
  std::vector<std::vector<std::string>> parsed_key_value_output;
  std::vector<std::string> key_value_output_names;

  for (const ov::Output<ov::Node>& output : model->outputs()) {
    const auto& names = output.get_names();

    for (const auto& name : names) {
      std::vector<std::string> parsed_name = ParseInputOutput(name);
      if (!parsed_name.empty() && parsed_name[0] == "present") {
        parsed_key_value_output.push_back(std::move(parsed_name));
        key_value_output_names.push_back(name);
        break;
      }
    }
  }

  return std::make_pair(parsed_key_value_output, key_value_output_names);
}

// Main function to extract KV tensors using dynamic pattern matching
//
// Example: Given input names ["input_ids", "attention_mask", "past_key_cross_0", "past_key_cross_1", "past_value_cross_0", "past_value_cross_1"]
//
//   parsed_key_value_input = [["past", "key", "cross", "0"], ["past", "key", "cross", "1"], ["past", "value", "cross", "0"], ["past", "value", "cross", "1"]]
//   key_value_input_names = ["past_key_cross_0", "past_key_cross_1", "past_value_cross_0", "past_value_cross_1"]
//   not_kv_inputs = ["input_ids", "attention_mask"]
std::tuple<std::vector<std::vector<std::string>>, std::vector<std::string>, std::vector<std::string>> ExtractInputKVTensors(
    const std::shared_ptr<ov::Model>& model) {
  std::vector<std::vector<std::string>> parsed_key_value_input;
  std::vector<std::string> not_kv_inputs;
  std::vector<std::string> kv_inputs;

  for (const ov::Output<ov::Node>& input : model->inputs()) {
    const auto& names = input.get_names();
    const std::string input_name = input.get_any_name();
    bool found = false;

    for (const auto& name : names) {
      std::vector<std::string> parsed_name = ParseInputOutput(name);

      bool has_layer = std::any_of(parsed_name.begin(), parsed_name.end(), [](const std::string& t) {
        return !t.empty() && std::all_of(t.begin(), t.end(), ::isdigit);
      });
      bool has_past_key_value = std::any_of(parsed_name.begin(), parsed_name.end(), [](const std::string& t) {
        return t == "past" || t == "key" || t == "value";
      });

      if (has_layer && has_past_key_value) {
        parsed_key_value_input.push_back(std::move(parsed_name));
        kv_inputs.push_back(name);
        found = true;
        break;  // one name confirmed KV — no need to check the rest
      }
    }

    if (!found) {
      not_kv_inputs.push_back(input_name);
    }
  }

  return std::make_tuple(parsed_key_value_input, not_kv_inputs, kv_inputs);  // outside the loop
}

// Updated PatchStatefulDecoder function
void PatchStatefulDecoder(std::shared_ptr<ov::Model> model, const bool should_add_kvcache_reorder) {
  // Use the dynamic pattern-based extraction logic
  auto [parsed_key_value_output, key_value_output_names] = ExtractOutputKVTensors(model);
  auto [parsed_key_value_input, not_kv_inputs, key_value_input_names] = ExtractInputKVTensors(model);

  OVEP_ENFORCE(!key_value_input_names.empty() && !key_value_output_names.empty(),
               "No key_value_input_names or key_value_output_names found");
  if (key_value_input_names.size() != key_value_output_names.size()) {
    const auto input_count = key_value_input_names.size();
    const auto output_count = key_value_output_names.size();
    OVEP_THROW("Found different sizes between key_value_input_names (",
               input_count,
               ") and key_value_output_names (",
               output_count,
               "). They couldn't be paired.");
  }

  std::map<std::string, std::string> input_output_map = MatchKVTensors(key_value_input_names, parsed_key_value_input, key_value_output_names, parsed_key_value_output);

  // By default, batch is the 0 - th but chatglm uses 1 - st dimension as batch
  // TODO(ryan): Deduce from a model via ordinal reshape(? ) and topology
  // batch_dim = 1 if config.model_type == "chatglm" and not hasattr(config, "rope_ratio") else 0
  auto batch_dim = 0;
  FuseCacheReorder(model, not_kv_inputs, key_value_input_names, batch_dim, should_add_kvcache_reorder);
  MakeStateful(model, input_output_map);
}

// Some other utility functions copied from OpenVINO GenAI
bool HasOpWithType(const std::shared_ptr<const ov::Model>& function, const std::string& type_name) {
  for (const auto& op : function->get_ops()) {
    if (op->get_type_name() == type_name) {
      return true;
    }
  }
  return false;
}

std::tuple<std::shared_ptr<ov::Node>, int64_t> FindLLMMatmul(const std::shared_ptr<ov::Model>& model) {
  auto last_node = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
  std::shared_ptr<ov::Node> matmul = ov::as_type_ptr<ov::op::v0::MatMul>(last_node);

  // In the case of PagedAttention, all tokens are moved to the batch dimension,
  // and slicing/gathering must be performed accordingly.
  const bool pa_based_model = HasOpWithType(model, "PagedAttentionExtension");
  int64_t slice_gather_dim = pa_based_model ? 0 : 1;

  // There are several patterns for MatMul we are looking for:
  // MatMul -> Result
  // MatMul -> Add -> Result
  // MatMul -> Transpose -> Result
  // MatMul -> Divide -> Tanh -> Multiply -> Result
  // MatMul -> Convert -> Result
  if (!matmul) {
    if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
    } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
      auto order_constant = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
      if (!order_constant) {
        return std::make_tuple(nullptr, slice_gather_dim);
      }
      auto order = order_constant->get_axis_vector_val();
      if (slice_gather_dim < 0 || static_cast<size_t>(slice_gather_dim) >= order.size()) {
        return std::make_tuple(nullptr, slice_gather_dim);
      }
      slice_gather_dim = order[static_cast<size_t>(slice_gather_dim)];
    } else if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(last_node)) {
      if (auto tanh = ov::as_type_ptr<ov::op::v0::Tanh>(multiply->input_value(0).get_node_shared_ptr())) {
        if (auto divide = ov::as_type_ptr<ov::op::v1::Divide>(tanh->input_value(0).get_node_shared_ptr())) {
          matmul = ov::as_type_ptr<ov::op::v0::MatMul>(divide->input_value(0).get_node_shared_ptr());
        }
      }
    } else if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(convert->input_value(0).get_node_shared_ptr());
    }
  }
  return std::make_tuple(matmul, slice_gather_dim);
}

void ApplySliceBeforeMatmulTransformation(std::shared_ptr<ov::Model> model) {
  std::shared_ptr<ov::Node> matmul = nullptr;
  int64_t slice_gather_dim = -1;
  std::tie(matmul, slice_gather_dim) = FindLLMMatmul(model);

  if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
    auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
    auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
    matmul->input(0).replace_source_output(slice);
  }
}

void UpdateConfig(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
  if (config.count(pair.first) == 0) {
    config.insert(pair);
  }
}

std::optional<ov::Any> PopOption(ov::AnyMap& config, const std::string& option_name) {
  if (auto it = config.find(option_name); it != config.end()) {
    std::optional<ov::Any> found = std::make_optional(it->second);
    config.erase(it);
    return found;
  }
  return std::nullopt;
}

void RenameKey(ov::AnyMap& config, const std::string& old_key, const std::string& new_key) {
  if (config.count(old_key) != 0) {
    auto opt_value = PopOption(config, old_key);
    config[new_key] = opt_value.value();
  }
}

KVAxesPosition GetKVAxesPos(std::shared_ptr<const ov::Model> model) {
  // Sequence length axis in key/values tensors. For most cases, the tensor shape is
  // [batch_size, num_kv_heads, seq_len, head_size]. Therefore, the sequence length axis
  // is usually at index 2, and the batch axis is at index 0.
  KVAxesPosition kv_pos{0u, 2u};

  // "ReadValue" node is KV cache representation in stateful model
  std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

  for (const auto& op : model->get_ops()) {
    // Check input size, as in LoRA adapters case it could be 0
    if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
      continue;
    }

    // Shape example: [-1,4,0,64]
    auto shape = op->get_input_partial_shape(0);

    for (int64_t i = 0; i < shape.rank().get_length(); i++) {
      // Find axis = 0. This would be sequence length axis.
      if (shape[i] == 0) {
        kv_pos.seq_len = i;
      } else if (shape[i].is_dynamic()) {
        // Dynamic axis is a batch
        kv_pos.batch = i;
      }
    }
    break;
  }

  return kv_pos;
}

void UpdateNPUConfig(ov::AnyMap& config, const KVAxesPosition& kv_pos, const KVDesc& kv_desc) {
  UpdateConfig(config, {"NPU_USE_NPUW", "YES"});
  UpdateConfig(config, {"NPUW_LLM", "YES"});

  UpdateConfig(config, {"NPUW_LLM_BATCH_DIM", kv_pos.batch});
  UpdateConfig(config, {"NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len});

  UpdateConfig(config, {"NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len});
  UpdateConfig(config, {"NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len});

  UpdateConfig(config, {"CACHE_MODE", "OPTIMIZE_SPEED"});

  RenameKey(config, "++PREFILL_CONFIG", "++NPUW_LLM_PREFILL_CONFIG");
  RenameKey(config, "++GENERATE_CONFIG", "++NPUW_LLM_GENERATE_CONFIG");
  RenameKey(config, "PREFILL_CONFIG", "NPUW_LLM_PREFILL_CONFIG");
  RenameKey(config, "PREFILL_HINT", "NPUW_LLM_PREFILL_HINT");
  RenameKey(config, "GENERATE_CONFIG", "NPUW_LLM_GENERATE_CONFIG");
  RenameKey(config, "GENERATE_HINT", "NPUW_LLM_GENERATE_HINT");
}

std::optional<uint32_t> PopIntAndCast(ov::AnyMap& config, const std::string& key) {
  auto anyopt = PopOption(config, key);
  if (anyopt.has_value()) {
    const auto any = anyopt.value();
    int64_t value;
    // NB: Integer value coming from python has int64_t datatype
    if (any.is<int64_t>()) {
      value = any.as<int64_t>();
    } else if (any.is<int>()) {
      value = any.as<int>();
    } else if (any.is<std::string>()) {
      value = std::stoll(any.as<std::string>());
    } else {
      OVEP_THROW("Failed to extract ", key, ". Type mismatch: expected types: int, int64_t, or string");
    }
    OVEP_ENFORCE(value >= 0, key, " cannot be negative!");
    constexpr uint32_t max_uint32 = std::numeric_limits<uint32_t>::max();
    OVEP_ENFORCE(value <= static_cast<int64_t>(max_uint32),
                 key,
                 " exceeds the maximum supported value of ",
                 max_uint32);
    return std::make_optional(static_cast<uint32_t>(value));
  }
  return std::nullopt;
}

bool IsStateful(const std::shared_ptr<ov::Model>& model) {
  for (auto&& ptr : model->get_ordered_ops()) {
    if (ov::is_type<ov::op::v3::ReadValue>(ptr) ||
        ov::is_type<ov::op::v6::ReadValue>(ptr) ||
        ov::is_type<ov::op::v3::Assign>(ptr) ||
        ov::is_type<ov::op::v6::Assign>(ptr)) {
      return true;
    }
  }
  return false;
}

bool HasTensor(const ov::CompiledModel& compiled_model, const std::string& tensor_name) {
  for (const auto& input : compiled_model.inputs()) {
    const auto& names = input.get_names();
    if (names.find(tensor_name) != names.end()) {
      return true;
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
