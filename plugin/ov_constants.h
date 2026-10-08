// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime/onnxruntime_c_api.h"
#include <unordered_map>
#include <string>

namespace onnxruntime {
namespace openvino_ep_plugin {

constexpr std::string_view env_string_debug{"ORT_OPENVINO_ENABLE_DEBUG"};
constexpr std::string_view env_string_accept_ops{"ORT_OPENVINO_ACCEPT_OPS"};
constexpr std::string_view env_string_reject_ops{"ORT_OPENVINO_REJECT_OPS"};
constexpr std::string_view env_string_disable_arena{"ORT_OPENVINO_DISABLE_ARENA"};
constexpr std::string_view env_string_cpu_va_for_igpu{"ORT_OPENVINO_CPU_VA_FOR_IGPU"};
constexpr std::string_view env_string_device_specific_allocators{"ORT_OPENVINO_DISABLE_DEVICE_SPECIFIC_ALLOCATORS"};
constexpr std::string_view env_string_enable_plugin_unloading{"ORT_OPENVINO_ENABLE_PLUGIN_UNLOADING"};
constexpr bool enable_mmap{true};

constexpr std::string_view key_ov_meta_device = "ov_meta_device";

const auto ep_gqa_value_tensor_layout_preference = std::unordered_map<OrtHardwareDeviceType, const char*>{
    {OrtHardwareDeviceType_CPU, "BNSH"},
    {OrtHardwareDeviceType_GPU, "BNSH"},
    {OrtHardwareDeviceType_NPU, "BNHS"},
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
