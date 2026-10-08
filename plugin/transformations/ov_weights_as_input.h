// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

namespace weights_as_inputs {
OrtStatus* TransformSharedWeightsToInputs(std::shared_ptr<ov::Model>& model, SharedContext& shared_context);
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
