// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime_abi {
///////////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct device_wrapper {
  std::string name_{};
  ort_execution_provider_t ort_execution_provider_;

  std::optional<OrtExecutionProviderDevicePolicy> ort_selection_policy_{};
  std::vector<Ort::ConstEpDevice> ep_device_;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(device_wrapper);
  OVEP_TESTS_DEFAULT_MOVE(device_wrapper);

  ///////////////////////////////////////////////////////////////////////////
  device_wrapper(std::string name = "")
      : name_(get_device_name(name)), ort_execution_provider_(name_ == "MLAS" ? ort_execution_provider_t::cpu : ort_execution_provider_t::ov) {
    auto policy_it = ort_device_policy_str_to_enum.find(name_);
    if (policy_it != ort_device_policy_str_to_enum.end()) {
      ort_selection_policy_ = policy_it->second;
    }

    if (ort_execution_provider_ != ort_execution_provider_t::cpu && ort_selection_policy_ == std::nullopt) {
      auto ep_devices = env_.GetEpDevices();

      std::unordered_map<std::string, std::vector<Ort::ConstEpDevice> > ep_device_map;
      for (const auto& device : ep_devices) {
        ep_device_map[device.EpName()].push_back(device);
      }

      // Parse metadevice from name_ if it contains ":"
      std::string metadevice;
      std::unordered_set<std::string> device_set;
      if (auto colon_pos = name_.find(':'); colon_pos != std::string::npos) {
        metadevice = name_.substr(0, colon_pos);
        std::string devices_str = name_.substr(colon_pos + 1);

        // Split comma-separated device list
        std::stringstream ss(devices_str);
        std::string device;
        while (std::getline(ss, device, ',')) {
          device_set.insert(device);
        }
      } else {
        device_set.insert(name_);
      }

      std::string expected_ep_name = "OpenVINOExecutionProvider";
      if (!metadevice.empty()) {
        expected_ep_name += "." + metadevice;
      }

      for (const auto& [ep_name, devices] : ep_device_map) {
        for (const auto& device : devices) {
          if (ep_name == expected_ep_name &&
              device_set.count(device.EpMetadata().GetValue("ov_device")) == 1) {
            ep_device_.emplace_back(device);
          }
        }
      }

      if (ep_device_.empty()) {
        OVEP_TESTS_LOG_WARN << " No EP device enumerated to be appended to session!";
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  std::string get_device_name(const std::string& name) {
    if (!name.empty()) {
      return name;
    }

    const auto device_name_env_var = ovep_tests::get_environment_variable("OVEP_TESTS_DEVICE_NAME");
    if (device_name_env_var.has_value()) {
      return *device_name_env_var;
    }

    return "MLAS";
  }

  ///////////////////////////////////////////////////////////////////////////
  OrtCompiledModelCompatibility ValidateCompatibility(const char* compatibility_info) const {
    return Ort::GetModelCompatibilityForEpDevices(ep_device_, compatibility_info);
  }

#ifdef _WIN32
  ///////////////////////////////////////////////////////////////////////////
  static LUID get_device_luid(const Ort::ConstEpDevice& ep_device) {
    auto luid_ull = std::stoull(ep_device.Device().Metadata().GetValue("LUID"));

    LUID luid{};
    luid.LowPart = static_cast<DWORD>(luid_ull & 0xFFFFFFFF);
    luid.HighPart = static_cast<LONG>((luid_ull >> 32) & 0xFFFFFFFF);

    return luid;
  }
#endif
};
}  // namespace onnxruntime_abi
}  // namespace ovep_tests
