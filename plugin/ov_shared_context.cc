// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ov_shared_context.h"
#include "ov_plugin_utils.h"

#include <gsl/narrow>

#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/core/type/element_type.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {
namespace {

uint32_t GetCpuVaCacheline(const std::string& device_name) {
  static std::mutex mutex;
  static std::unordered_map<std::string, uint32_t> cacheline_by_device;

  std::lock_guard lock(mutex);
  if (const auto it = cacheline_by_device.find(device_name);
      it != cacheline_by_device.end()) {
    return it->second;
  }

  uint32_t cacheline = 0;
  try {
    cacheline = WeakSingleton<ov::Core>::Get()->get_property(
        device_name, ov::intel_gpu::cacheline_size);
  } catch (const ov::Exception&) {
  }

  cacheline_by_device.emplace(device_name, cacheline);
  return cacheline;
}

// CPU_VA zero-copy (default) vs per-weight copy for iGPU weights; ORT_OPENVINO_CPU_VA_FOR_IGPU=0 opts out.
bool UseCpuVa(const std::string& dev_name, const Ort::Logger& logger) {
  static const bool cpu_va_enabled = features::cpu_va_for_igpu.is_enabled();
  static const bool banner_logged = [&] {
    ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_INFO, "[CPU_VA] iGPU weight loading mode: %s",
                 cpu_va_enabled ? "CPU_VA zero-copy (OV 2026.4+)" : "per-weight copy (ORT_OPENVINO_CPU_VA_FOR_IGPU=0)");
    return true;
  }();
  (void)banner_logged;
  if (!cpu_va_enabled) {
    return false;
  }

  const auto cacheline = GetCpuVaCacheline(dev_name);
  if (cacheline == 0) {
    static const bool warned = [&] {
      ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING,
                   "[CPU_VA] Device '%s' reports cacheline_size=0 (CPU_VA import unsupported); "
                   "using per-weight copy.",
                   dev_name.c_str());
      return true;
    }();
    (void)warned;
    return false;
  }
  return true;
}

std::string TryGetRemoteDeviceName(const std::optional<ov::RemoteContext>& remote_context) {
  if (!remote_context) {
    return {};
  }

  return remote_context->get_device_name();
}

std::optional<ov::device::Type> TryGetDeviceType(const std::optional<ov::RemoteContext>& remote_context) {
  std::string device_name;
  device_name = TryGetRemoteDeviceName(remote_context);
  if (device_name.empty()) {
    return std::nullopt;
  }

  try {
    auto ov_core = WeakSingleton<ov::Core>::Get();
    return ov_core->get_property(device_name, ov::device::type);
  } catch (const ov::Exception&) {
    return std::nullopt;
  }
}

}  // namespace

SharedContext::SharedContext(const std::filesystem::path& bin_path)
    : bin_path_(bin_path),
      bin_manager_(bin_path_),
      weight_file_manager_(WeightFileManager::Get()) {}

// Weights file handling
SharedContext::WeightsFile::WeightsFile(const std::filesystem::path& filename)
    : file_path_(filename), file_size_(GetOrCreateCpuMappedTensor().get_byte_size()) {}

ov::Tensor SharedContext::WeightsFile::GetOrCreateCpuMappedTensor() const {
  std::lock_guard<std::mutex> lock(mapped_tensor_mutex_);
  if (!mapped_tensor_) {
    auto tensor = ov::read_tensor_data(file_path_);
    OVEP_ENFORCE(tensor, "Error: Failed to memory map weight file at ", ToUtf8String(file_path_));
    mapped_tensor_.emplace(std::move(tensor));
  }

  return *mapped_tensor_;
}

const void* SharedContext::WeightsFile::TryGetOrCreateDeviceMapping(std::optional<ov::RemoteContext>& remote_context,
                                                                    std::optional<ov::device::Type> device_type,
                                                                    const Ort::Logger& logger) {
  const std::string dev_name = TryGetRemoteDeviceName(remote_context);

  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = imported_device_tensors_.emplace(dev_name, MappingContainer{});
  if (inserted) {
    if (dev_name == "NPU") {
      OVEP_ENFORCE(remote_context, "Error: Remote context is required for NPU device.");
      auto npu_context = remote_context->as<ov::intel_npu::level_zero::ZeroContext>();
      auto gpu_it = imported_device_tensors_.end();
      for (auto candidate = imported_device_tensors_.begin(); candidate != imported_device_tensors_.end(); ++candidate) {
        if (candidate->first.starts_with("GPU") && candidate->second.ptr_ != nullptr) {
          gpu_it = candidate;
          break;
        }
      }
      if (gpu_it != imported_device_tensors_.end()) {
        // A GPU has already mapped this shared weight file; import the same host pointer so both
        // devices share one allocation.
        auto&& l0_tensor = npu_context.create_tensor(ov::element::Type_t::u8, {SizeAligned()},
                                                     const_cast<void*>(gpu_it->second.ptr_),
                                                     ov::intel_npu::MemType::CPU_VA);
        it->second = MappingContainer{.ptr_ = l0_tensor.get(), .tensor_ = l0_tensor};
      } else {
        auto&& l0_tensor =
            npu_context.create_tensor(ov::element::Type_t::u8, {Size()}, ov::intel_npu::FileDescriptor(file_path_));
        it->second = MappingContainer{.ptr_ = l0_tensor.get(), .tensor_ = l0_tensor};
      }
    } else if (dev_name.starts_with("GPU")) {
      if (device_type == ov::device::Type::DISCRETE) {
        // dGPU isn't host-addressable; return nullptr so the caller copies each weight.
        return nullptr;
      }
      OVEP_ENFORCE(remote_context, "Error: Remote context is required for GPU device.");

      // iGPU without CPU_VA leaves an empty mapping so the caller copies each weight.
      if (UseCpuVa(dev_name, logger)) {
        auto mapped_tensor = GetOrCreateCpuMappedTensor();
        const void* mapped_ptr = static_cast<const ov::Tensor&>(mapped_tensor).data();
        it->second = MappingContainer{.ptr_ = mapped_ptr, .tensor_ = std::move(mapped_tensor)};
      }
    } else if (dev_name.empty()) {
      // CPU/virtual device case, reuse a lazy memory map of the whole file.
      auto mapped_tensor = GetOrCreateCpuMappedTensor();
      const void* mapped_ptr = static_cast<const ov::Tensor&>(mapped_tensor).data();
      it->second = MappingContainer{.ptr_ = mapped_ptr, .tensor_ = std::move(mapped_tensor)};
    }
  }

  return it->second.ptr_;
}

std::optional<ov::Tensor> SharedContext::WeightsFile::TryCreateDeviceTensor(
    std::optional<ov::RemoteContext>& remote_context, std::optional<ov::device::Type> device_type,
    const ov::element::Type& element_type, const ov::Shape& dimensions, size_t data_offset, size_t size,
    const Ort::Logger& logger) {
  const auto* base = static_cast<const uint8_t*>(TryGetOrCreateDeviceMapping(remote_context, device_type, logger));
  if (base == nullptr) {
    // No zero-copy mapping (e.g. dGPU); caller copies the weight into a device tensor.
    return std::nullopt;
  }

  void* data_ptr = const_cast<uint8_t*>(base) + data_offset;
  const std::string dev_name = remote_context ? remote_context->get_device_name() : std::string{};
  const bool is_igpu =
      remote_context && dev_name.rfind("GPU", 0) == 0 && device_type == ov::device::Type::INTEGRATED;

  if (is_igpu && UseCpuVa(dev_name, logger)) {
    // CPU_VA needs a cacheline-aligned address and cacheline-multiple size (OV create_hostbuffer_impl);
    // the mmap base is page-aligned, so per-weight alignment is decided by the file offset.
    const uint32_t cacheline = GetCpuVaCacheline(dev_name);
    const bool aligned = cacheline != 0 && reinterpret_cast<uintptr_t>(data_ptr) % cacheline == 0;
    if (!aligned) {
      ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_INFO,
                   "[CPU_VA] Unaligned weight in '%s' offset=%zu addr=%p not %u-byte aligned; "
                   "copying instead of zero-copy.",
                   ToUtf8String(file_path_).c_str(), data_offset, data_ptr, cacheline);
      return std::nullopt;
    }
    const auto aligned_size = gsl::narrow<int64_t>(AlignUp(size, cacheline));
    const ov::AnyMap cpu_va_params = {
        {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::CPU_VA},
        {ov::intel_gpu::cpu_va_access.name(), ov::intel_gpu::AccessMode::READ},
        {ov::intel_gpu::cpu_va.name(), static_cast<ov::intel_gpu::gpu_handle_param>(data_ptr)},
        {ov::intel_gpu::cpu_va_size.name(), aligned_size},
    };
    // Import can still throw on drivers that don't support it; degrade to the copy path.
    try {
      return remote_context->create_tensor(element_type, dimensions, cpu_va_params);
    } catch (const ov::Exception& ex) {
      ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_INFO,
                   "[CPU_VA] create_tensor import failed on '%s' (%s); copying instead of zero-copy.",
                   dev_name.c_str(), ex.what());
      return std::nullopt;
    }
  }

  // NPU / iGPU CPU_VA fallback / CPU: base is host-accessible, so view the weight directly.
  return ov::Tensor(element_type, dimensions, data_ptr);
}

void SharedContext::LoadTensorFromFile(
    Metadata::Value& value,
    const std::filesystem::path& model_dir,
    std::optional<ov::RemoteContext>& remote_context,
    std::optional<ov::device::Type> device_type,
    const ov::element::Type& element_type,
    const ov::Shape& dimensions,
    const Ort::Logger& logger) {
  const auto weights_location = model_dir / value.serialized.location;
  auto& weights_file = weight_files_[weights_location];
  if (!weights_file) {
    weights_file = weight_file_manager_->GetOrCreateWeightsFile(weights_location);
  }

  OVEP_ENFORCE(IsWithinBounds(value.serialized.data_offset, value.serialized.size, weights_file->Size()),
               "File offset + size outside of external initializer file");

  const std::string device_name = TryGetRemoteDeviceName(remote_context);

  ov::Tensor tensor;
  if (auto device_tensor = weights_file->TryCreateDeviceTensor(remote_context, device_type, element_type, dimensions,
                                                               value.serialized.data_offset, value.serialized.size,
                                                               logger)) {
    tensor = std::move(*device_tensor);
  } else {
    // No device mapping for this weight; create a device tensor and copy the weight into it.
    OVEP_ENFORCE(remote_context, "Remote context required when device memory mapping unavailable");
    auto remote_tensor = remote_context->create_tensor(element_type, dimensions);

    auto mapped_tensor = weights_file->GetOrCreateCpuMappedTensor();
    const uint8_t* host_ptr = static_cast<const uint8_t*>(static_cast<const ov::Tensor&>(mapped_tensor).data());
    const void* data_ptr = host_ptr + value.serialized.data_offset;
    ov::Tensor host_view(element_type, dimensions, const_cast<void*>(data_ptr));
    // The file bounds check used the metadata size; validate the actual copy extent before reading.
    OVEP_ENFORCE(host_view.get_byte_size() == value.serialized.size, "Tensor size mismatch");
    remote_tensor.copy_from(host_view);
    tensor = std::move(remote_tensor);
  }

  OVEP_ENFORCE(tensor.get_byte_size() == value.serialized.size, "Tensor size mismatch");
  value.device_tensors[device_name] = std::make_shared<const ov::Tensor>(std::move(tensor));
}

void SharedContext::SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const std::filesystem::path& model_dir,
                                                   const Ort::Logger& logger) {
  auto&& compiled_model = ir.get_compiled_model();
  std::optional<ov::RemoteContext> opt_remote_ctx;
  try {
    opt_remote_ctx = compiled_model.get_context();
  } catch (ov::Exception&) {
    // CPU may not have a remote context.
  }

  const auto device_type = TryGetDeviceType(opt_remote_ctx);
  const std::string device_name = TryGetRemoteDeviceName(opt_remote_ctx);

  std::unique_lock<std::shared_mutex> ul(mutex_);
  for (const auto& input : compiled_model.inputs()) {
    const std::string tensor_name = *input.get_names().begin();

    auto it = metadata_.find(tensor_name);
    if (it == metadata_.end()) continue;  // No shared weight for this tensor
    auto& value = it->second;

    auto& tensor = value.device_tensors[device_name];
    if (!tensor) {
      LoadTensorFromFile(value, model_dir, opt_remote_ctx, device_type, input.get_element_type(), input.get_shape(),
                         logger);
    }

    ir.set_tensor(tensor_name, *tensor);
  }
}

void SharedContext::Serialize(std::ostream& stream) {
  bin_manager_.Serialize(stream, shared_from_this());
}

void SharedContext::Deserialize(OwnedBuffer buffer) {
  bin_manager_.Deserialize(std::move(buffer), shared_from_this());
}

void SharedContext::Serialize() {
  bin_manager_.Serialize(shared_from_this());
}

void SharedContext::Deserialize() {
  bin_manager_.Deserialize(shared_from_this());
}

std::string SharedContext::SerializeToString() {
  return bin_manager_.SerializeToString(shared_from_this());
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
