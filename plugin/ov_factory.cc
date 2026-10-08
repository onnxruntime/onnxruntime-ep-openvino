// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <set>
#include <ranges>
#include <format>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <gsl/narrow>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/onnxruntime_ep_device_ep_metadata_keys.h"
#include "onnxruntime/onnxruntime_env_config_keys.h"

#include "ort_api_version.h"
#include "onnxruntime/onnxruntime_experimental_cxx_api.h"
#include "ov_bin_manager.h"
#include "ov_common_utils.h"
#include "ov_factory.h"
#include "ov_data_transfer.h"
#include "ov_external_resource_importer.h"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "weak_singleton.h"
#include "ep_arena.h"

using namespace onnxruntime::openvino_ep_plugin;
using ov_core_singleton = WeakSingleton<ov::Core>;
using onnx_fe_manager_singleton = WeakSingleton<ov::frontend::FrontEndManager>;

// Unload OpenVINO plugins when the instance of OvPluginUnloader is destroyed
struct OpenVINOEpPluginFactory::OvPluginUnloader : public WeakSingleton<OvPluginUnloader> {
  ~OvPluginUnloader() {
    if (!features::enable_plugin_unloading.is_enabled()) {
      return;
    }

    auto ov_core = ov_core_singleton::Get();
    for (auto ov_device : OvDevice::known_physical_devices_) {
      try {
        ov_core->unload_plugin(ov_device);
      } catch (const ov::Exception&) {
        // Plugin is not loaded
      }
    }
  }
};

static OrtMemoryInfoDeviceType OrtHwDevTypeToMemHwDevType(OrtHardwareDeviceType device_type) {
  switch (device_type) {
    case OrtHardwareDeviceType_CPU:
      return OrtMemoryInfoDeviceType_CPU;
    case OrtHardwareDeviceType_GPU:
      return OrtMemoryInfoDeviceType_GPU;
    case OrtHardwareDeviceType_NPU:
      return OrtMemoryInfoDeviceType_NPU;
    default:
      OVEP_THROW("Unknown OrtHardwareDeviceType: ", device_type);
  }
}

OrtStatus* OpenVINOEpPluginFactory::ParseHardwareDeviceTypeFilter(const char* value,
                                                                  std::set<OrtHardwareDeviceType>& out_filter) {
  out_filter.clear();
  if (value == nullptr || value[0] == '\0') {
    return nullptr;
  }

  static const std::unordered_map<std::string, OrtHardwareDeviceType> name_to_type = {
      {"CPU", OrtHardwareDeviceType_CPU},
      {"GPU", OrtHardwareDeviceType_GPU},
      {"NPU", OrtHardwareDeviceType_NPU},
  };

  for (auto&& token : std::ranges::split_view(std::string_view(value), ';')) {
    auto token_str = onnxruntime::openvino_ep::utils::TrimWhitespace(std::string(token.begin(), token.end()));
    if (token_str.empty()) {
      continue;  // Skip empty segments (e.g., "CPU;;GPU" or a trailing ';')
    }

    std::string upper_token = token_str;
    std::ranges::transform(upper_token, upper_token.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    auto it = name_to_type.find(upper_token);
    if (it == name_to_type.end()) {
      out_filter.clear();
      std::string message =
          std::format("Invalid value '{}' for '{}': expected a semicolon-separated list of CPU, GPU, NPU.",
                      token_str, hardware_device_type_env_key_);
      return Ort::Status(message.c_str(), ORT_INVALID_ARGUMENT).release();
    }

    out_filter.insert(it->second);
  }

  return nullptr;
}

OpenVINOEpPluginFactory::OpenVINOEpPluginFactory(const std::string& ov_metadevice_name, bool allow_virtual_devices,
                                                 std::set<OrtHardwareDeviceType> hardware_device_type_filter,
                                                 std::shared_ptr<ov::Core> core, const OrtLogger* logger)
    : OrtEpFactory{},
      ep_name_(ov_metadevice_name.empty() ? provider_name_ : std::string(provider_name_) + "." + ov_metadevice_name),
      device_type_(ov_metadevice_name),
      ov_core_(std::move(core)),
      onnx_fe_manager_(onnx_fe_manager_singleton::Get()),
      ov_plugin_unloader_(OvPluginUnloader::Get()),
      raw_logger_(logger),
      logger_(logger),
      allow_virtual_devices_(allow_virtual_devices),
      hardware_device_type_filter_(std::move(hardware_device_type_filter)) {
  OrtEpFactory::GetName = OVEP_API_ENTRY(OpenVINOEpPluginFactory, GetName);
  OrtEpFactory::GetVendor = OVEP_API_ENTRY(OpenVINOEpPluginFactory, GetVendor);
  OrtEpFactory::GetSupportedDevices = OVEP_API_ENTRY(OpenVINOEpPluginFactory, GetSupportedDevices);
  OrtEpFactory::CreateEp = OVEP_API_ENTRY(OpenVINOEpPluginFactory, CreateEp);
  OrtEpFactory::ReleaseEp = OVEP_API_ENTRY(OpenVINOEpPluginFactory, ReleaseEp);
  OrtEpFactory::CreateAllocator = OVEP_API_ENTRY(OpenVINOEpPluginFactory, CreateAllocator);
  OrtEpFactory::ReleaseAllocator = OVEP_API_ENTRY(OpenVINOEpPluginFactory, ReleaseAllocator);
  OrtEpFactory::CreateDataTransfer = OVEP_API_ENTRY(OpenVINOEpPluginFactory, CreateDataTransfer);
  OrtEpFactory::IsStreamAware = OVEP_API_ENTRY(OpenVINOEpPluginFactory, IsStreamAware);
  OrtEpFactory::GetVersion = OVEP_API_ENTRY(OpenVINOEpPluginFactory, GetVersion);
  OrtEpFactory::GetVendorId = OVEP_API_ENTRY(OpenVINOEpPluginFactory, GetVendorId);
  OrtEpFactory::ValidateCompiledModelCompatibilityInfo = OVEP_API_ENTRY(OpenVINOEpPluginFactory, ValidateCompiledModelCompatibilityInfo);

  OrtVersionGate<24>(static_cast<OrtEpFactory*>(this), [this](auto* factory) {
    factory->CreateExternalResourceImporterForDevice = OVEP_API_ENTRY(OpenVINOEpPluginFactory, CreateExternalResourceImporterForDevice);
  });

  ort_version_supported = ort_runtime_api_version;
}

const std::vector<std::string>& OpenVINOEpPluginFactory::GetOvMetaDevices() {
  static std::vector<std::string> virtual_devices = [ov_core = ov_core_singleton::Get()] {
    std::vector<std::string> supported_virtual_devices{};
    for (const auto& meta_device : OvDevice::known_meta_devices_) {
      try {
        ov_core->get_property(meta_device, ov::supported_properties);
        supported_virtual_devices.push_back(meta_device);
      } catch (ov::Exception&) {
        // meta device isn't supported.
      }
    }
    return supported_virtual_devices;
  }();

  return virtual_devices;
}

OrtStatus* OpenVINOEpPluginFactory::GetSupportedVirtualDevices(OrtEpDevice** ep_devices,
                                                               size_t max_ep_devices,
                                                               size_t* p_num_ep_devices) {
  // Virtual devices are always NPU-typed; skip entirely if NPU is excluded by the filter.
  if (!hardware_device_type_filter_.empty() && !hardware_device_type_filter_.contains(OrtHardwareDeviceType_NPU)) {
    return nullptr;
  }

  if (!IsMetaDeviceFactory()) {
    size_t& num_ep_devices = *p_num_ep_devices;

    Ort::KeyValuePairs hw_metadata{};
    hw_metadata.Add(kOrtHardwareDevice_MetadataKey_IsVirtual, "1");

    // No need for a runtime version check before calling CreateHardwareDevice or ReleaseHardwareDevice because these funtions will only be called
    // when allow_virtual_devices is set which is only available in ORT 1.24
    for (const auto& [device_id, platform] : npu_device_id_to_platform) {
      if (num_ep_devices < max_ep_devices) {
        OrtHardwareDevice* hw_device = nullptr;
        OVEP_RETURN_IF_ERROR(ep_api.CreateHardwareDevice(OrtHardwareDeviceType_NPU, vendor_id_, device_id, vendor_, hw_metadata, &hw_device));

        hw_devices_.emplace_back(hw_device, [this](OrtHardwareDevice* hw_device) {
          ep_api.ReleaseHardwareDevice(hw_device);
        });

        Ort::KeyValuePairs ep_metadata{};
        ep_metadata.Add("ov_device", "NPU");
        ep_metadata.Add("npu_platform", platform.c_str());
        ep_metadata.Add(kOrtEpDevice_EpMetadataKey_WeightlessSupport, "external_only");

        OrtEpDevice* ep_device = nullptr;
        OVEP_RETURN_IF_ERROR(ep_api.CreateEpDevice(this, hw_device, ep_metadata, /* ep_options */ {}, &ep_device));

        ep_devices[num_ep_devices++] = ep_device;
      }
    }
  }

  return nullptr;
}

std::string OpenVINOEpPluginFactory::MatchOvDevice(const Ort::ConstHardwareDevice& hw_device) {
  if (hw_device.VendorId() != vendor_id_) {
    return {};
  }

  // Excluded device types must never reach GetOvDevices()/get_property() below, since that call
  // is what triggers OpenVINO to load the corresponding device plugin.
  if (!hardware_device_type_filter_.empty() && !hardware_device_type_filter_.contains(hw_device.Type())) {
    return {};
  }

  auto device_it = ort_to_ov_device_name_.find(hw_device.Type());
  if (device_it == ort_to_ov_device_name_.end()) {
    return {};
  }

  const auto& ov_device_type = device_it->second;
  auto filtered_devices = GetOvDevices(ov_device_type);

  auto matched_device = filtered_devices.begin();
  if (filtered_devices.size() > 1 && hw_device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
    matched_device = std::find_if(filtered_devices.begin(), filtered_devices.end(), [&](const std::string& ov_device) {
      try {
        const std::string device_id_str = ov_core_->get_property(ov_device, "GPU_DEVICE_ID").as<std::string>();
        return hw_device.DeviceId() == static_cast<uint32_t>(std::stoul(device_id_str, nullptr, 0));
      } catch (ov::Exception&) {
        return false;
      }
    });
  }

  if (matched_device == filtered_devices.end()) {
    return {};
  }

  return *matched_device;
}

OrtStatus* OpenVINOEpPluginFactory::GetSupportedPhysicalDevices(const OrtHardwareDevice* const* devices,
                                                                size_t num_devices,
                                                                OrtEpDevice** ep_devices,
                                                                size_t max_ep_devices,
                                                                size_t* p_num_ep_devices) {
  size_t& num_ep_devices = *p_num_ep_devices;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    Ort::ConstHardwareDevice device{devices[i]};
    auto ov_device_name = MatchOvDevice(device);
    if (ov_device_name.empty()) {
      continue;
    }

    // these can be returned as nullptr if you have nothing to add.
    Ort::KeyValuePairs ep_metadata;

    ep_metadata.Add(ov_device_key_, ov_device_name.c_str());

    // Add OS driver version if available
    uint32_t driver_version = [ov_core = ov_core_singleton::Get(), &ov_device_name]() -> uint32_t {
      try {
        if (ov_device_name.starts_with("NPU")) {
          return ov_core->get_property(ov_device_name, ov::intel_npu::driver_version.name()).as<uint32_t>();
        } else {
          return 0;
        }
      } catch (ov::Exception&) {
        return 0;
      }
    }();

    if (driver_version != 0) {
      ep_metadata.Add(kOrtEpDevice_EpMetadataKey_OSDriverVersion, std::to_string(driver_version).c_str());
    }

    const bool supports_weightless = !IsMetaDeviceFactory() && DeviceSupportsWeightless(ov_device_name);
    ep_metadata.Add(kOrtEpDevice_EpMetadataKey_WeightlessSupport, supports_weightless ? "external_only" : "none");

    // For meta devices, add the meta device type to the metadata
    if (IsMetaDeviceFactory()) {
      ep_metadata.Add(std::string(key_ov_meta_device).data(), device_type_.c_str());
    } else {
      ep_metadata.Add(kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout, ep_gqa_value_tensor_layout_preference.at(device.Type()));
    }

    // Add all read-only device properties to the metadata
    auto get_property_as_string = [&](const std::string& prop_name) -> std::optional<std::string> {
      try {
        return ov_core_->get_property(ov_device_name, prop_name).as<std::string>();
      } catch (const ov::Exception&) {
        return std::nullopt;
      }
    };

    try {
      const auto supported_props = ov_core_->get_property(ov_device_name, ov::supported_properties);
      for (const auto& prop : supported_props) {
        // skip SUPPORTED_PROPERTIES as it's redundant
        if (!prop.is_mutable() && prop != ov::supported_properties.name()) {
          if (auto value_str = get_property_as_string(prop)) {
            ort_api.AddKeyValuePair(ep_metadata, prop.c_str(), value_str->c_str());
          }
        }
      }
    } catch (const ov::Exception&) {
      // Failed to iterate properties, continue without them
    }

    // Create EP device
    Ort::EpDevice ep_device{*this, device, ep_metadata.GetConst()};

    if (!IsMetaDeviceFactory()) {
      OrtMemoryInfoDeviceType mem_info_dev_type = OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
      int mem_info_dev_id = 0;
      std::string allocator_name{OVRTAllocator::allocator_prefix.substr(0, OVRTAllocator::allocator_prefix.size() - 1)};  // base allocator name without device suffix or separator

      if (!features::disable_device_specific_allocators.is_enabled()) {
        OrtVersionGate<27>([&]() {
          // In ORT v27+, we use a new memory info device type for OpenVINO devices, and assign unique IDs to each device.
          mem_info_dev_type = OrtHwDevTypeToMemHwDevType(device.Type());
          // CPU memory info gets ID 0, device memory infos get unique IDs.
          mem_info_dev_id = mem_info_dev_type == OrtMemoryInfoDeviceType_CPU ? 0 : gsl::narrow<int>(mem_infos.size() + 1);
          allocator_name += OVRTAllocator::allocator_prefix.back() + ov_device_name;
        });
      }

      auto add_allocator_info = [&](OrtDeviceMemoryType mem_type, OrtAllocatorType allocator_type) -> Ort::ConstMemoryInfo {
        auto& mem_info = mem_infos.emplace_back(allocator_name.c_str(),
                                                mem_info_dev_type,
                                                vendor_id_,
                                                mem_info_dev_id,
                                                mem_type,
                                                OVRTAllocator::alignment_,
                                                allocator_type);
        Ort::ThrowOnError(ort_api.GetEpApi()->EpDevice_AddAllocatorInfo(ep_device, mem_info.GetConst()));
        return mem_info.GetConst();
      };

      OpenVINOEpPlugin::MemoryInfos memory_infos{};
      // Only register the HOST_ACCESSIBLE memory info. We intentionally do not register a device-typed
      // DEFAULT memory info; memory_infos.default_ stays null and the plugin's default device resolves
      // to this host-accessible memory (see OpenVINOEpPlugin::GetDefaultMemoryDevice).
      memory_infos.cpu_input_ = memory_infos.cpu_output_ = add_allocator_info(OrtDeviceMemoryType_HOST_ACCESSIBLE, OrtDeviceAllocator);

      memory_infos_by_ov_device_[ov_device_name] = std::move(memory_infos);
    }

    OrtEpDevice* released_ep_device = ep_device.release();
    ep_devices[num_ep_devices++] = released_ep_device;
  }

  // Unloads unused OpenVINO plugins after all factories have enumerated devices
  ov_plugin_unloader_.reset();
  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                                        size_t num_devices,
                                                        OrtEpDevice** ep_devices,
                                                        size_t max_ep_devices,
                                                        size_t* p_num_ep_devices) {
  *p_num_ep_devices = 0;

  // only enumerate virtual devices if allow virtual devices is set
  if (allow_virtual_devices_) {
    return GetSupportedVirtualDevices(ep_devices, max_ep_devices, p_num_ep_devices);
  } else {
    return GetSupportedPhysicalDevices(devices, num_devices, ep_devices, max_ep_devices, p_num_ep_devices);
  }
}

OrtStatus* OpenVINOEpPluginFactory::CreateEp(const OrtHardwareDevice* const* devices,
                                             const OrtKeyValuePairs* const* ep_metadata,
                                             size_t num_devices,
                                             const OrtSessionOptions* session_options,
                                             const OrtLogger* logger,
                                             OrtEp** ep) {
  *ep = nullptr;

  Ort::Logger ort_logger{logger};

  // Check if no devices are provided
  if (num_devices == 0) {
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "No devices provided");
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "No devices provided to CreateEp");
  }

  std::unordered_set<std::string_view> unique_ov_devices;
  std::vector<std::string_view> ordered_unique_ov_devices;
  for (size_t i = 0; i < num_devices; ++i) {
    const char* ov_device_char = ort_api.GetKeyValue(ep_metadata[i], ov_device_key_);
    std::string_view ov_device = ov_device_char ? ov_device_char : std::string_view{};
    if (ov_device.empty()) {
      ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "Ep device missing expected metadata key");
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  std::format("Ep device missing expected metadata key {}", ov_device_key_).c_str());
    }

    // Add to ordered_unique only if not already present
    if (unique_ov_devices.insert(ov_device).second) {
      ordered_unique_ov_devices.push_back(ov_device);
    }
  }

  bool is_meta_device_factory = IsMetaDeviceFactory();
  if (ordered_unique_ov_devices.size() > 1 && !is_meta_device_factory) {
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_WARNING,
                std::format("OpenVINO EP factory '{}' is not a meta device but {} OpenVINO devices were specified. Using first ov_device only: {}",
                            ep_name_, unique_ov_devices.size(), ordered_unique_ov_devices.at(0))
                    .c_str());
    ordered_unique_ov_devices.resize(1);  // Use only the first device if not a meta device factory
  }

  OvDevice ov_device = is_meta_device_factory
                           ? OvDevice(device_type_, ordered_unique_ov_devices)
                           : OvDevice(ordered_unique_ov_devices[0], allow_virtual_devices_);
  ORT_CXX_LOGF(ort_logger, ORT_LOGGING_LEVEL_INFO, "resolved ov_device='%s'", ov_device.c_str());

  OpenVINOEpPluginOptions options(ep_name_, ort_logger, Ort::ConstSessionOptions{session_options});

  if (options.ep_ctx_.enable_ && ov_device.IsMetaDevice()) {
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "EPContext model generation/caching is not supported for OpenVINO meta-devices");
    return ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        std::format("EPContext model generation/caching is not supported for OpenVINO meta-device '{}'. "
                    "Select the physical device directly via the '{}' EP instead.",
                    ov_device.c_str(), provider_name_)
            .c_str());
  }

  if (allow_virtual_devices_) {
    if (num_devices > 1) {
      ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "Only 1 device supported for compilation");
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, std::format("Only 1 device supported for compilation, found {}", num_devices).c_str());
    }

    if (ov_device != "NPU") {
      ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "Incorrect ov_device for compilation, only NPU is supported");
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, std::format("Incorrect ov_device for compilation, only NPU is supported").c_str());
    }

    // Add NPU platform to provider options for virtual devices
    auto device_id = ort_api.HardwareDevice_DeviceId(devices[0]);
    auto itr = npu_device_id_to_platform.find(device_id);
    if (itr == npu_device_id_to_platform.end()) {
      ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, "Unsupported NPU device ID");
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, std::format("Unsupported NPU device ID: 0x{:X}", device_id).c_str());
    } else {
      options.provider_options_.SetLoadConfigOption("NPU", "NPU_PLATFORM", itr->second);
    }

    // Set NPU_COMPILER_TYPE to PLUGIN to disable fallback to compiler in driver
    options.provider_options_.SetLoadConfigOption("NPU", "NPU_COMPILER_TYPE", "PLUGIN");
  }

  // Meta-device factories (HETERO/AUTO) leave memory_infos default-constructed (all null);
  // the provider then falls through to ORT's default-device heuristic.
  OpenVINOEpPlugin::MemoryInfos memory_infos;
  if (auto it = memory_infos_by_ov_device_.find(static_cast<std::string>(ov_device));
      it != memory_infos_by_ov_device_.end()) {
    memory_infos = it->second;
  }

  // Create a new OpenVINO execution provider with this factory's device type
  auto ov_ep = std::make_unique<OpenVINOEpPlugin>(ep_name_, std::move(options), *logger, ov_device, ov_core_, onnx_fe_manager_, memory_infos);

  OrtVersionGate<28>(ov_ep.get(), [&](auto ov_ep) {
    Ort::Experimental::EpContextConfig cfg{Ort::ConstSessionOptions{session_options}};
    OrtWriteNamedBufferFunc write_fn = nullptr;
    void* write_state = nullptr;
    cfg.GetWriteFunc(write_fn, write_state);
    OrtReadNamedBufferFunc read_fn = nullptr;
    void* read_state = nullptr;
    cfg.GetReadFunc(read_fn, read_state);
    ov_ep->SetRawEpContextDataCallbacks(write_fn, write_state, read_fn, read_state);
  });

  *ep = ov_ep.release();
  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::CreateAllocator(const OrtMemoryInfo* memory_info,
                                                    const OrtKeyValuePairs* allocator_options,
                                                    OrtAllocator** allocator) {
  Ort::ConstMemoryInfo mem_info{memory_info};
  auto mem_info_name = mem_info.GetAllocatorName();

  const std::string_view shared_prefix = OVRTAllocator::allocator_prefix;

  if (!mem_info_name.starts_with(shared_prefix)) {
    // ORT < v27: the allocator name has no device suffix, so we cannot resolve the
    // OpenVINO device here. Fall back to the plain passthrough OVRTAllocator on
    // host-accessible ("CPU") memory (legacy behavior). The pooling arena is only
    // used on v27+, where the device is identifiable from the allocator name.
    *allocator = new OVRTAllocator(*ov_core_, *memory_info, OvDevice{"CPU"}, logger_);
    return nullptr;
  }

  // ORT v27+: allocator name embeds the device suffix (e.g. "OpenVINO_shared_GPU.0").
  const std::string ov_device_name{mem_info_name.substr(shared_prefix.size())};
  auto ep_allocator = std::make_unique<OVRTAllocator>(*ov_core_, *memory_info, OvDevice{ov_device_name}, logger_);

  if (features::disable_arena.is_enabled()) {
    ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING,
                 "%s set: using passthrough OVRTAllocator for '%s' (arena pooling disabled)",
                 env_string_disable_arena.data(), ov_device_name.c_str());
    *allocator = ep_allocator.release();
    return nullptr;
  }

  std::unique_ptr<ArenaAllocator> arena_allocator;
  OVEP_RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(ep_allocator), allocator_options, ort_api, *raw_logger_, arena_allocator));
  *allocator = arena_allocator.release();
  return nullptr;
}

void OpenVINOEpPluginFactory::ReleaseAllocator(OrtAllocator* allocator) noexcept {
  delete static_cast<BaseAllocator*>(allocator);
}

OrtStatus* OpenVINOEpPluginFactory::CreateDataTransfer(OrtDataTransferImpl** data_transfer) {
  *data_transfer = new OvDataTransfer(logger_);
  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::ValidateCompiledModelCompatibilityInfo(
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) {
  *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  if (!compatibility_info || compatibility_info[0] == '\0') {
    return nullptr;
  }

  // Parse the JSON envelope produced by GetCompiledModelCompatibilityInfo
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(compatibility_info);
  } catch (const nlohmann::json::parse_error& e) {
    ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "Failed to parse compatibility info JSON: %s", e.what());
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  std::string ovep_compat_ver = j.value("ovep_compat_ver", "");

  if (ovep_compat_ver == "1.0") {
    // Version 1.0: check bin version + forward OV string to OpenVINO
    std::string model_bin_ver = j.value("ovep_bin_ver", "");
    if (!BinManager::IsBinVersionSupported(model_bin_ver)) {
      ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING,
                   "Model bin version '%s' is newer than supported '%s'",
                   model_bin_ver.c_str(), bin_format::GetBinCompatVersion().c_str());
      *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      return nullptr;
    }
  } else {
    // Unknown version — cannot validate yet, return NOT_APPLICABLE
    // Future versions of the string may validate additional EP requirements
    return nullptr;
  }

  std::string ov_compat_string = j.value("ov_compat_string", "");
  if (ov_compat_string.empty()) {
    return nullptr;
  }

  std::string target_device;
  for (size_t i = 0; i < num_devices; ++i) {
    Ort::ConstHardwareDevice hw_device{devices[i]};
    target_device = MatchOvDevice(hw_device);
    if (!target_device.empty()) {
      break;
    }
  }

  if (target_device.empty()) {
    return nullptr;
  }

  ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_INFO, "ValidateCompatibility: device='%s' ov_string='%s'",
               target_device.c_str(), ov_compat_string.c_str());

  ov::CompatibilityCheck ov_compat = ov::CompatibilityCheck::NOT_APPLICABLE;
  try {
    ov_compat = ov_core_->get_property(target_device,
                                       ov::compatibility_check,
                                       {{ov::runtime_requirements.name(), ov_compat_string}});
  } catch (const ov::Exception& e) {
    ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "ov::compatibility_check threw: %s", e.what());
    return nullptr;
  }

  switch (ov_compat) {
    case ov::CompatibilityCheck::SUPPORTED:
      *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
      break;
    case ov::CompatibilityCheck::UNSUPPORTED:
      *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      break;
    case ov::CompatibilityCheck::NOT_APPLICABLE:
    default:
      *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
      break;
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPluginFactory::CreateExternalResourceImporterForDevice(const OrtEpDevice* ep_device,
                                                                            OrtExternalResourceImporterImpl** importer) {
  *importer = nullptr;

#ifdef _WIN32
  if (IsMetaDeviceFactory()) {
    return nullptr;
  }

  std::string_view ov_device = Ort::ConstEpDevice{ep_device}.EpMetadata().GetValue(ov_device_key_);
  if (ov_device != "NPU") {
    return nullptr;
  }

  auto resource_importer =
      std::make_unique<OvExternalResourceImporter>(*ov_core_, Ort::ConstEpDevice{ep_device}, ov_device, logger_);
  *importer = resource_importer.release();
#else
  (void)ep_device;

  ORT_CXX_API_THROW("External resource importer is not supported for Linux", ORT_NOT_IMPLEMENTED);
#endif

  return nullptr;
}

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base, const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  OVEP_RETURN_IF_ERROR(InitCxxApiOrStatus(ort_api_base));

  Ort::Logger trace_logger{default_logger};

  bool allow_virtual_devices = false;
  OrtVersionGate<24>([&allow_virtual_devices, default_logger]() {
    auto env_configs = Ort::GetEnvConfigEntries();
    const char* config_value = env_configs.GetValue(kOrtEnvAllowVirtualDevices);
    allow_virtual_devices = config_value != nullptr && strcmp(config_value, "1") == 0;
  });

  std::set<OrtHardwareDeviceType> hardware_device_type_filter;
  OrtStatus* hardware_device_type_status = nullptr;
  OrtVersionGate<24>([&hardware_device_type_filter, &hardware_device_type_status]() {
    auto env_configs = Ort::GetEnvConfigEntries();
    const char* config_value = env_configs.GetValue(OpenVINOEpPluginFactory::hardware_device_type_env_key_);
    hardware_device_type_status =
        OpenVINOEpPluginFactory::ParseHardwareDeviceTypeFilter(config_value, hardware_device_type_filter);
  });
  if (hardware_device_type_status != nullptr) {
    return hardware_device_type_status;
  }

  std::vector<std::string> supported_factories{""};
  std::shared_ptr<ov::Core> ov_core;
  {
    OVEP_TRACE(trace_logger, "ov::Core() + GetOvMetaDevices()");
    // Get available devices from OpenVINO
    ov_core = ov_core_singleton::Get();

    // Add meta-device factories
    const auto& meta_devices = OpenVINOEpPluginFactory::GetOvMetaDevices();
    supported_factories.insert(supported_factories.end(), meta_devices.begin(), meta_devices.end());
  }

  const size_t required_factories = supported_factories.size();
  if (max_factories < required_factories) {
    return Ort::Status(std::format("Not enough space to return EP factories. Need at least {} factories.", required_factories).c_str(), ORT_INVALID_ARGUMENT).release();
  }

  size_t factory_index = 0;
  for (const auto& device_name : supported_factories) {
    // Create a factory for this specific device
    factories[factory_index++] = new OpenVINOEpPluginFactory(device_name, allow_virtual_devices, hardware_device_type_filter, ov_core, default_logger);
  }

  *num_factories = factory_index;
  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<OpenVINOEpPluginFactory*>(factory);
  return nullptr;
}
}
