// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "transformations/ov_weights_as_input.h"

#include "ov_utils.h"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

using ov_const_set_t = std::set<std::shared_ptr<ov::op::v0::Constant>, std::owner_less<std::shared_ptr<ov::op::v0::Constant>>>;

static std::unordered_set<std::shared_ptr<ov::op::v0::Constant>> FindDequantScaleZPConstants(std::shared_ptr<ov::Model>& model) {
  // Match a dequantize subgraph and capture the scale/zero-point source Constants directly. The
  // scale and zero-point reach the Multiply/Subtract through optional Convert/Reshape (and, for the
  // unaligned MatMulNBits zero-point, a Slice) inserted by the frontend; modelling those as optional
  // pattern nodes binds the backing Constant without a manual graph walk.
  using namespace ov::pass::pattern;

  // Zero point: Constant -> Convert -> (optional Slice) -> (optional Reshape) -> Subtract.
  auto zero_point = wrap_type<ov::op::v0::Constant>();
  auto zp_convert = wrap_type<ov::op::v0::Convert>({zero_point});
  auto zp_slice = optional<ov::op::v8::Slice>({zp_convert, any_input(), any_input(), any_input(), any_input()});
  auto zp_reshape = optional<ov::op::v1::Reshape>({zp_slice, any_input()});
  auto subtract = wrap_type<ov::op::v1::Subtract>({any_input(), zp_reshape});

  // Scale: Constant -> (optional Convert) -> (optional Reshape) -> Multiply.
  // NOTE: only a single Convert is modelled. A future op emitting two Converts here (e.g. mixed
  // scale/zp float widths) would miss the scale Constant; add another optional Convert if that arises.
  auto scale = wrap_type<ov::op::v0::Constant>();
  auto scale_convert = optional<ov::op::v0::Convert>({scale});
  auto scale_reshape = optional<ov::op::v1::Reshape>({scale_convert, any_input()});
  auto multiply = wrap_type<ov::op::v1::Multiply>({subtract, scale_reshape});

  auto matcher = std::make_shared<ov::pass::pattern::Matcher>(multiply, "dequantize_pattern");

  std::unordered_set<std::shared_ptr<ov::op::v0::Constant>> dequant_scale_zp_consts;
  for (auto&& node : model->get_ordered_ops()) {
    // Only need to run the pattern on Multiply nodes
    if (!std::dynamic_pointer_cast<ov::op::v1::Multiply>(node)) continue;
    if (!matcher->match(node->output(0))) continue;

    auto& pattern_map = matcher->get_pattern_value_map();

    // The pattern binds scale/zero_point to their source Constants, so exclude them directly.
    if (auto it = pattern_map.find(scale); it != pattern_map.end()) {
      dequant_scale_zp_consts.emplace(ov::as_type_ptr<ov::op::v0::Constant>(it->second.get_node_shared_ptr()));
    }
    if (auto it = pattern_map.find(zero_point); it != pattern_map.end()) {
      dequant_scale_zp_consts.emplace(ov::as_type_ptr<ov::op::v0::Constant>(it->second.get_node_shared_ptr()));
    }
  }
  return dequant_scale_zp_consts;
}

std::shared_ptr<ov::Model> ConvertConstantsToInputs(std::shared_ptr<ov::Model> model, SharedContext& shared_context) {
  // Zero points and scales of dequantize operations will not be converted to inputs
  auto dequant_scale_zp_consts = FindDequantScaleZPConstants(model);

  // Convert each constant to a parameter that are NOT dequantization parameters
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_parameters;
  for (auto&& node : model->get_ordered_ops()) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    // Only convert constants that are known shared weights.
    if (!constant || !shared_context.IsSharedWeight(constant->get_friendly_name())) continue;
    // Only convert constant if it's not a dequantization scale or zero_point
    if (dequant_scale_zp_consts.contains(constant)) continue;

    // Create parameter with same shape and type as constant
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        constant->get_element_type(),
        constant->get_shape());
    parameter->set_friendly_name(constant->get_friendly_name());

    // Replace all uses of constant with the new parameter
    constant->output(0).replace(parameter->output(0));

    new_parameters.push_back(std::move(parameter));
  }

  // Get existing parameters and add new ones
  auto original_params = model->get_parameters();
  original_params.insert(original_params.end(), new_parameters.begin(), new_parameters.end());

  // Create new model with additional parameters
  auto new_model = std::make_shared<ov::Model>(
      model->get_results(),
      original_params,
      model->get_friendly_name() + "_weights_as_inputs");

  return new_model;
}

OrtStatus* weights_as_inputs::TransformSharedWeightsToInputs(std::shared_ptr<ov::Model>& model, SharedContext& shared_context) {
  try {
    model = ConvertConstantsToInputs(model, shared_context);
    return nullptr;  // Success
  } catch (const std::exception& e) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: ", e.what()).c_str(),
                       ORT_RUNTIME_EXCEPTION);
  } catch (...) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: unknown error").c_str(), ORT_RUNTIME_EXCEPTION);
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
