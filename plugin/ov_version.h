// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include "openvino/core/version.hpp"

#define OV_VERSION(MAJOR, MINOR) (((MAJOR) << 32) | (MINOR))
#define CURRENT_OV_VERSION OV_VERSION(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR)

namespace onnxruntime {
namespace openvino_ep_plugin {

struct OpenVINOVersion {
  OpenVINOVersion() = default;
  constexpr OpenVINOVersion(const uint32_t major, const uint32_t minor)
      : version_((static_cast<uint64_t>(major) << 32) | minor) {}
  OpenVINOVersion(const std::string& version_str) {
    size_t dot_pos = version_str.find('.');
    try {
      if (dot_pos != std::string::npos) {
        auto major = static_cast<uint32_t>(std::stoi(version_str.substr(0, dot_pos)));
        auto minor = static_cast<uint32_t>(std::stoi(version_str.substr(dot_pos + 1)));
        version_ = (static_cast<uint64_t>(major) << 32) | minor;
      }
    } catch (const std::exception&) {
      // If parsing fails, keep version_ as 0
      version_ = 0;
    }
  }

  static constexpr OpenVINOVersion GetBuiltVersion() {
    return OpenVINOVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR);
  }

  constexpr uint32_t major_version() const { return static_cast<uint32_t>(version_ >> 32); }
  constexpr uint32_t minor_version() const { return static_cast<uint32_t>(version_ & 0xFFFFFFFF); }

  constexpr bool operator==(const OpenVINOVersion& other) const {
    return version_ == other.version_;
  }

  constexpr bool operator<(const OpenVINOVersion& other) const {
    return version_ < other.version_;
  }

  constexpr bool operator>=(const OpenVINOVersion& other) const {
    return version_ >= other.version_;
  }

 private:
  uint64_t version_{0};
};

static inline std::string to_string(const OpenVINOVersion& version) {
  return std::to_string(version.major_version()) + "." + std::to_string(version.minor_version());
}

static_assert(OpenVINOVersion::GetBuiltVersion() >= OpenVINOVersion(2026, 4),
              "OpenVINO version 2026.4 or higher is required.");

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
