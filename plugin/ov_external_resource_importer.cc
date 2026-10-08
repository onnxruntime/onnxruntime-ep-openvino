// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ov_external_resource_importer.h"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include <format>
#include <limits>

namespace onnxruntime {
namespace openvino_ep_plugin {

namespace {

constexpr size_t kNpuPageAlignment = 4096;

}  // namespace

OvExternalResourceImporter::OvExternalResourceImporter(ov::Core& core, Ort::ConstEpDevice ep_device,
                                                       std::string_view ov_device, Ort::Logger logger)
    : OrtExternalResourceImporterImpl{}, ep_device_(ep_device), logger_(logger) {
  try {
    remote_ctx_ = core.get_default_context(OvDevice{ov_device});
  } catch (const ov::Exception&) {
    // No remote context available for this device; leave remote_ctx_ empty. CanImportMemory
    // will report false for everything and ImportMemory will fail with ORT_EP_FAIL.
  }

  OrtExternalResourceImporterImpl::CanImportMemory = OVEP_API_ENTRY(OvExternalResourceImporter, CanImportMemory);
  OrtExternalResourceImporterImpl::ImportMemory = OVEP_API_ENTRY(OvExternalResourceImporter, ImportMemory);
  OrtExternalResourceImporterImpl::ReleaseMemory =
      OVEP_API_ENTRY_TEARDOWN(OvExternalResourceImporter, ReleaseMemory);
  OrtExternalResourceImporterImpl::CreateTensorFromMemory =
      OVEP_API_ENTRY(OvExternalResourceImporter, CreateTensorFromMemory);
  OrtExternalResourceImporterImpl::Release = OVEP_API_ENTRY_TEARDOWN(OvExternalResourceImporter, Release);

  ort_version_supported = ort_runtime_api_version;
}

bool OvExternalResourceImporter::CanImportMemory(OrtExternalMemoryHandleType handle_type) const {
  return remote_ctx_.has_value() &&
         (handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP ||
          handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);
}

OrtStatus* OvExternalResourceImporter::ImportMemory(const OrtExternalMemoryDescriptor* mem_desc,
                                                    OrtExternalMemoryHandle** mem_handle) {
  OVEP_RETURN_STATUS_IF(!CanImportMemory(mem_desc->handle_type), ORT_NOT_IMPLEMENTED,
                        "Unsupported memory handle type");
  OVEP_RETURN_STATUS_IF(mem_desc->native_handle == nullptr || mem_desc->size_bytes == 0, ORT_INVALID_ARGUMENT,
                        "native_handle must be non-null and size_bytes must be non-zero");
  OVEP_RETURN_STATUS_IF(mem_desc->offset_bytes >= mem_desc->size_bytes, ORT_INVALID_ARGUMENT,
                        "Memory descriptor offset exceeds total size");

  return mem_desc->handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION
             ? ImportCpuVa(*mem_desc, mem_handle)
             : ImportD3D12Heap(*mem_desc, mem_handle);
}

OrtStatus* OvExternalResourceImporter::ImportNpuBuffer(const OrtExternalMemoryDescriptor& mem_desc,
                                                       OrtExternalMemoryHandle** mem_handle, size_t alloc_bytes,
                                                       ov::intel_npu::MemType mem_type, const char* error_context) {
  OVEP_RETURN_STATUS_IF(!remote_ctx_, ORT_EP_FAIL, "No NPU remote context available");

  try {
    auto npu_context = remote_ctx_->as<ov::intel_npu::level_zero::ZeroContext>();
    auto ov_tensor = npu_context.create_tensor(ov::element::u8, {alloc_bytes},
                                               const_cast<void*>(mem_desc.native_handle), mem_type,
                                               ov::intel_npu::TensorType::BINDED);

    void* mapped_ptr = static_cast<uint8_t*>(ov_tensor.get()) + mem_desc.offset_bytes;
    auto ov_mem_handle = std::make_unique<OvExternalMemoryHandle>(mem_desc, static_cast<const OrtEpDevice*>(ep_device_), std::move(ov_tensor), mapped_ptr);
    *mem_handle = ov_mem_handle.release();
  } catch (const ov::Exception& ex) {
    return OVEP_MAKE_STATUS(std::format("{}: {}", error_context, ex.what()).c_str(), ORT_EP_FAIL);
  }
  return nullptr;
}

OrtStatus* OvExternalResourceImporter::ImportD3D12Heap(const OrtExternalMemoryDescriptor& mem_desc,
                                                       OrtExternalMemoryHandle** mem_handle) {
  // Note: There are known issues with importing heaps with nt handle on NPU with drivers older than 1004720.
  return ImportNpuBuffer(mem_desc, mem_handle, mem_desc.size_bytes, ov::intel_npu::MemType::SHARED_BUF,
                         "Failed to import D3D12 heap memory");
}

OrtStatus* OvExternalResourceImporter::ImportCpuVa(const OrtExternalMemoryDescriptor& mem_desc,
                                                   OrtExternalMemoryHandle** mem_handle) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(mem_desc.native_handle);
  OVEP_RETURN_STATUS_IF(
      addr % kNpuPageAlignment != 0, ORT_INVALID_ARGUMENT,
      "CPU_VA import requires a page-aligned address (4096 bytes). Allocate the buffer via "
      "ortvalue_from_shape_and_type_for_memory_info using this EP's HOST_ACCESSIBLE OrtMemoryInfo.");

  OVEP_RETURN_STATUS_IF(mem_desc.size_bytes > std::numeric_limits<size_t>::max() - (kNpuPageAlignment - 1),
                        ORT_INVALID_ARGUMENT, "CPU_VA memory size is too large");
  return ImportNpuBuffer(mem_desc, mem_handle, AlignUp(mem_desc.size_bytes, kNpuPageAlignment),
                         ov::intel_npu::MemType::CPU_VA, "Failed to import CPU_VA memory");
}

void OvExternalResourceImporter::ReleaseMemory(OrtExternalMemoryHandle* handle) {
  if (handle && handle->Release) {
    handle->Release(handle);
  }
}

void OvExternalResourceImporter::Release() {
  // The importer is being asked to delete itself. It's strange, but ORT seem to expect this in 1.24.
  delete this;
}

OrtStatus* OvExternalResourceImporter::CreateTensorFromMemory(const OrtExternalMemoryHandle* mem_handle,
                                                              const OrtExternalTensorDescriptor* tensor_desc,
                                                              OrtValue** value) {
  OVEP_RETURN_STATUS_IF(tensor_desc->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, ORT_INVALID_ARGUMENT,
                        "String tensors cannot be created from external memory");

  const auto* ov_mem_handle = static_cast<const OvExternalMemoryHandle*>(mem_handle);
  size_t usable_bytes = ov_mem_handle->UsableBytes();

  OVEP_RETURN_STATUS_IF(tensor_desc->offset_bytes >= usable_bytes, ORT_INVALID_ARGUMENT,
                        "Tensor descriptor offset exceeds available memory size");

  const size_t remaining_bytes = usable_bytes - tensor_desc->offset_bytes;

  void* ptr = static_cast<uint8_t*>(ov_mem_handle->mapped_ptr_) + tensor_desc->offset_bytes;

  // ORT calculates the tensor storage size, including overflow handling, and validates it against this byte count.
  auto tensor = Ort::Value::CreateTensor(ep_device_.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE), ptr,
                                         remaining_bytes, tensor_desc->shape, tensor_desc->rank,
                                         tensor_desc->element_type);
  *value = tensor.release();

  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
