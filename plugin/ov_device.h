// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <set>
#include <algorithm>
#include <cctype>

namespace onnxruntime {
namespace openvino_ep_plugin {

struct OvDevice {
  explicit OvDevice(std::string_view device_type, bool is_virtual = false) : name_(device_type), is_virtual_(is_virtual) {}
  explicit OvDevice(std::string_view meta_device_type, const std::vector<std::string_view>& devices)
      : name_(BuildMetaDeviceString(meta_device_type, devices)) {}

 private:
  static std::string BuildMetaDeviceString(std::string_view meta_device_type, const std::vector<std::string_view>& devices) {
    OVEP_ENFORCE(OvDevice::known_meta_devices_.contains(std::string(meta_device_type)), "Unknown meta device type: ", meta_device_type);

    std::string device_string(meta_device_type);
    device_string += ":";
    bool prepend_comma = false;
    for (const auto& device : devices) {
      if (prepend_comma) {
        device_string += ",";
      }
      device_string += device;
      prepend_comma = true;
    }
    return device_string;
  }

 public:
  operator const std::string&() const { return name_; }
  const char* c_str() const { return name_.c_str(); }

  bool operator==(std::string_view other) const { return name_ == other; }
  bool operator!=(std::string_view other) const { return name_ != other; }
  bool operator==(const OvDevice& other) const { return name_ == other.name_ && is_virtual_ == other.is_virtual_; }
  bool operator!=(const OvDevice& other) const { return name_ != other.name_ || is_virtual_ != other.is_virtual_; }

  bool IsVirtual() const { return is_virtual_; }
  bool IsMetaDevice() const {
    return std::any_of(known_meta_devices_.begin(), known_meta_devices_.end(),
                       [this](const std::string& meta_device) {
                         return name_ == meta_device || name_.starts_with(meta_device + ":");
                       });
  }
  bool IsDeviceType(std::string_view device_type) const {
    if (IsMetaDevice() || !known_physical_devices_.contains(std::string(device_type))) {
      return false;
    }

    return name_ == device_type ||
           (name_.starts_with(device_type) && name_.size() > device_type.size() && name_[device_type.size()] == '.');
  }

  inline static const std::set<std::string> known_meta_devices_ = {"AUTO"};
  inline static const std::set<std::string> known_physical_devices_ = {"CPU", "GPU", "NPU"};

  bool Contains(std::string_view substring) const {
    return name_.find(substring) != std::string::npos;
  }

 private:
  std::string name_;
  bool is_virtual_{false};
};

// For OV 2026.4, only NPU can produce weightless blobs.
inline bool DeviceSupportsWeightless(std::string_view ov_device) {
  return ov_device.starts_with("NPU");
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
