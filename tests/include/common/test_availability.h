// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <string_view>
#include <vector>

#include "env.h"

namespace ovep_tests {
///////////////////////////////////////////////////////////////////////
inline std::vector<std::string> get_physical_device_names(std::string_view device_name) {
  if (device_name == "MLAS") {
    return {};
  }

  const auto colon_pos = device_name.find(':');
  if (colon_pos == std::string_view::npos) {
    return {std::string(device_name)};
  }

  std::vector<std::string> physical_devices;
  std::string_view devices = device_name.substr(colon_pos + 1);
  while (!devices.empty()) {
    const auto comma_pos = devices.find(',');
    const auto physical_device = devices.substr(0, comma_pos);
    if (!physical_device.empty()) {
      physical_devices.emplace_back(physical_device);
    }
    if (comma_pos == std::string_view::npos) {
      break;
    }
    devices.remove_prefix(comma_pos + 1);
  }
  return physical_devices;
}

///////////////////////////////////////////////////////////////////////
inline bool is_openvino_device_available(
    const std::vector<Ort::ConstEpDevice>& ep_devices,
    std::string_view device_name) {
  const auto physical_devices = get_physical_device_names(device_name);
  if (physical_devices.empty()) {
    return device_name == "MLAS";
  }

  const auto colon_pos = device_name.find(':');
  const std::string expected_ep_name =
      colon_pos == std::string_view::npos
          ? "OpenVINOExecutionProvider"
          : "OpenVINOExecutionProvider." + std::string(device_name.substr(0, colon_pos));

  return std::all_of(physical_devices.begin(), physical_devices.end(), [&](const std::string& physical_device) {
    return std::any_of(ep_devices.begin(), ep_devices.end(), [&](const Ort::ConstEpDevice& ep_device) {
      if (ep_device.EpName() == nullptr || expected_ep_name != ep_device.EpName()) {
        return false;
      }
      return ep_device.EpMetadata().GetValue("ov_device") == physical_device;
    });
  });
}

///////////////////////////////////////////////////////////////////////
inline std::vector<std::string> get_unavailable_openvino_devices(
    const std::vector<Ort::ConstEpDevice>& ep_devices,
    const std::vector<std::string>& required_devices) {
  std::vector<std::string> unavailable_devices;
  for (const auto& device : required_devices) {
    if (!is_openvino_device_available(ep_devices, device) &&
        std::find(unavailable_devices.begin(), unavailable_devices.end(), device) == unavailable_devices.end()) {
      unavailable_devices.push_back(device);
    }
  }
  return unavailable_devices;
}

///////////////////////////////////////////////////////////////////////
}  // namespace ovep_tests
