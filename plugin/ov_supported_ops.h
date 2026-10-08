// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <set>

#include "ov_version.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace onnxruntime {
namespace openvino_ep_plugin {

struct SupportedOps {
  bool
  IsOpSupported(std::string_view op_type, std::string_view domain) const {
    return ops_.find(std::make_pair(op_type, domain)) != ops_.end();
  }

  bool IsOpSupported(const Ort::ConstNode& node) const {
    return IsOpSupported(node.GetOperatorType(), node.GetDomain());
  }

  static SupportedOps& Get() {
    static SupportedOps instance(OpenVINOVersion::GetBuiltVersion());
    return instance;
  }

 private:
  SupportedOps() = default;
  SupportedOps(const OpenVINOVersion& ov_version);
  OpenVINOVersion version_;
  std::set<std::pair<std::string_view, std::string_view>> ops_;
  std::vector<std::pair<std::string, std::string>> extra_ops_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
