// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <unordered_set>
#include <filesystem>

#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "openvino/runtime/core.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

struct EpContextSessionConfig {
  bool enable_{false};
  bool embed_{false};
  bool share_{false};
  bool stop_{false};
  std::filesystem::path path_{};
  bool weightless_{false};
  std::filesystem::path source_model_path_{};
  std::span<const uint8_t> source_model_buffer_{};
};

using ConfigMap = std::map<std::string, ov::AnyMap>;
using ReshapeMap = std::map<std::string, ov::PartialShape>;

struct OpenVINOEpProviderOptions {
  ConfigMap load_config_{};                      // JSON config map to load custom OV parameters.
  bool enable_causallm_{false};                  // Enables Causal LM Compilation for ORT GenAI OVEP Pass
  bool hint_enable_gqa_{false};                  // Opt-in support for com.microsoft::GroupQueryAttention.
  ReshapeMap reshape_input_{};                   // Used for reshaping the OV graph inputs shape at runtime.
  std::filesystem::path dump_subgraphs_path_{};  // Dump output path; non-empty means dump_subgraphs is enabled.

  static const std::unordered_set<std::string>& GetValidProviderKeys() {
    static const std::unordered_set<std::string> valid_keys = {
        "load_config", "enable_causallm", "hint_enable_gqa", "reshape_input", "dump_subgraphs"};
    return valid_keys;
  }

  auto GetDeviceConfig(const std::string& ov_device) const {
    std::string device_name = ov_device;

    // Get meta device name, For eg AUTO when ov_device is AUTO:CPU,GPU
    auto pos = ov_device.find_first_of(':');
    if (pos != std::string::npos) {
      device_name = ov_device.substr(0, pos);
    }

    auto itr = load_config_.find(device_name);
    return itr != load_config_.end() ? itr->second : ov::AnyMap{};
  }

  void SetLoadConfigOption(const std::string& device, const std::string& key, const std::string& val) {
    auto& device_map = load_config_[device];
    device_map[key] = val;
  }

  void SetLoadConfigOption(const std::string& device, const ov::AnyMap& map) {
    auto& device_map = load_config_[device];
    for (const auto& [key, val] : map) {
      device_map[key] = val;
    }
  }
};

struct OpenVINOEpPluginOptions {
  OpenVINOEpPluginOptions(std::string ep_name, const Ort::Logger& logger, Ort::ConstSessionOptions session_options);

  EpContextSessionConfig ep_ctx_;
  OpenVINOEpProviderOptions provider_options_;
  bool debug_enabled_{false};

 private:
  std::string ep_name_;
  static EpContextSessionConfig ParseEpContextOptions(Ort::ConstSessionOptions session_options);
  void ParseProviderOptions(const Ort::Logger& logger, Ort::ConstSessionOptions session_options);
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
