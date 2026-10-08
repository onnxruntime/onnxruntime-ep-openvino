// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

#include "ov_provider.h"
#include "ov_custom_allocator.h"
#include "openvino/runtime/core.hpp"
#include "openvino/frontend/manager.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

class OpenVINOEpPluginFactory : public OrtEpFactory {
 public:
  OpenVINOEpPluginFactory(const std::string& ov_device, bool allow_virtual_devices,
                          std::set<OrtHardwareDeviceType> hardware_device_type_filter,
                          std::shared_ptr<ov::Core> ov_core, const OrtLogger* logger);
  virtual ~OpenVINOEpPluginFactory() = default;

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPluginFactory)

  Ort::Logger GetLogger() const { return logger_; }

  // Returns available OV device names for the given device type.
  // Single device: returns ["GPU"]. Multiple devices: returns ["GPU.0", "GPU.1", ...].
  std::vector<std::string> GetOvDevices(const std::string& device_type) {
    std::vector<std::string> available_devices;
    std::vector<std::string> device_ids;
    try {
      device_ids = ov_core_->get_property(device_type, ov::available_devices);
    } catch (...) {
      // Any failure (e.g. plugin not loaded, invalid env) — return empty list.
    }

    if (device_ids.size() == 1) {
      available_devices.push_back(device_type);
    } else if (device_ids.size() > 1) {
      std::transform(device_ids.begin(), device_ids.end(), std::back_inserter(available_devices),
                     [&device_type](const std::string& id) { return device_type + '.' + id; });
    }

    return available_devices;
  }

  static const std::vector<std::string>& GetOvMetaDevices();

  const char* GetName() const noexcept {
    return ep_name_.c_str();
  }

  const char* GetVendor() const noexcept {
    return vendor_;
  }

  bool IsStreamAware() const noexcept {
    return false;
  }

  const char* GetVersion() const noexcept {
    return OVEP_PLUGIN_VER_TAGGED;
  }

  uint32_t GetVendorId() const noexcept {
    return vendor_id_;
  }

  OrtStatus* GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices);
  OrtStatus* GetSupportedPhysicalDevices(const OrtHardwareDevice* const* devices,
                                         size_t num_devices,
                                         OrtEpDevice** ep_devices,
                                         size_t max_ep_devices,
                                         size_t* p_num_ep_devices);
  OrtStatus* GetSupportedVirtualDevices(
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
      size_t* p_num_ep_devices);

  OrtStatus* CreateEp(const OrtHardwareDevice* const* devices,
                      const OrtKeyValuePairs* const* ep_metadata,
                      size_t num_devices,
                      const OrtSessionOptions* session_options,
                      const OrtLogger* logger,
                      OrtEp** ep);

  void ReleaseEp(OrtEp* ep) {
    delete static_cast<OpenVINOEpPlugin*>(ep);
  }

  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator);
  void ReleaseAllocator(OrtAllocator* allocator) noexcept;
  OrtStatus* CreateDataTransfer(OrtDataTransferImpl** data_transfer);
  OrtStatus* CreateExternalResourceImporterForDevice(const OrtEpDevice* ep_device,
                                                     OrtExternalResourceImporterImpl** importer);

  OrtStatus* ValidateCompiledModelCompatibilityInfo(const OrtHardwareDevice* const* devices,
                                                    size_t num_devices,
                                                    const char* compatibility_info,
                                                    OrtCompiledModelCompatibility* model_compatibility);

  bool IsMetaDeviceFactory() const {
    return OvDevice::known_meta_devices_.find(device_type_) != OvDevice::known_meta_devices_.end();
  }

  // Constants
  static constexpr const char* vendor_ = "Intel";
  static constexpr uint32_t vendor_id_{0x8086};  // Intel's PCI vendor ID
  static constexpr const char* ov_device_key_ = "ov_device";
  static constexpr const char* provider_name_ = OpenVINOEpPlugin::provider_name_;
  static constexpr const char* hardware_device_type_env_key_ = "ep.openvinoexecutionprovider.hardware_device_type";

  std::string MatchOvDevice(const Ort::ConstHardwareDevice& hw_device);
  static OrtStatus* ParseHardwareDeviceTypeFilter(const char* value, std::set<OrtHardwareDeviceType>& out_filter);

 private:
  struct OvPluginUnloader;

  std::string ep_name_;
  const OrtApi& ort_api{Ort::GetApi()};
  const OrtEpApi& ep_api{Ort::GetEpApi()};
  std::string device_type_;
  std::shared_ptr<ov::Core> ov_core_;
  std::shared_ptr<ov::frontend::FrontEndManager> onnx_fe_manager_;
  std::vector<Ort::MemoryInfo> mem_infos;
  // Meta-device factories (HETERO/AUTO) don't register per-physical-device allocators, so
  // they leave this empty and the provider falls through to ORT's default-device heuristic.
  std::map<std::string, OpenVINOEpPlugin::MemoryInfos, std::less<>> memory_infos_by_ov_device_;
  std::shared_ptr<OvPluginUnloader> ov_plugin_unloader_;
  const OrtLogger* raw_logger_;
  Ort::Logger logger_;

  bool allow_virtual_devices_{false};
  std::set<OrtHardwareDeviceType> hardware_device_type_filter_;
  using unique_hw_device = std::unique_ptr<OrtHardwareDevice, std::function<void(OrtHardwareDevice*)>>;
  std::vector<unique_hw_device> hw_devices_;

  inline static const std::map<OrtHardwareDeviceType, std::string> ort_to_ov_device_name_ = {
      {OrtHardwareDeviceType_CPU, "CPU"},
      {OrtHardwareDeviceType_GPU, "GPU"},
      {OrtHardwareDeviceType_NPU, "NPU"},
  };

  inline static const std::unordered_map<uint32_t, std::string> npu_device_id_to_platform = {
      {0x643E, "4000"},  // LunarLake
      {0xB03E, "5010"},  // PantherLake
      {0xFD3E, "5020"},  // WildcatLake
      {0xD71D, "6010"},  // NovaLake
  };
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
