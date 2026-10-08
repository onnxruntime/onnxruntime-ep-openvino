// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime/onnxruntime_c_api.h"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "ov_provider.h"
#include "ov_utils.h"
#include "ov_device.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

// Derived handle. ORT frees this via OrtExternalMemoryHandle::Release (interop_api.cc), NOT via
// OrtExternalResourceImporterImpl::ReleaseMemory, which ORT never calls.
struct OvExternalMemoryHandle : public OrtExternalMemoryHandle {
  OvExternalMemoryHandle(const OrtExternalMemoryDescriptor& mem_desc, const OrtEpDevice* device,
                         ov::Tensor&& tensor, void* mapped_ptr)
      : OrtExternalMemoryHandle{}, ov_tensor_(std::move(tensor)), mapped_ptr_(mapped_ptr) {
    descriptor = mem_desc;
    version = mem_desc.version;
    ep_device = device;
    Release = OVEP_API_ENTRY_TEARDOWN(OvExternalMemoryHandle, ReleaseSelf);
  }

  ~OvExternalMemoryHandle() = default;
  OVEP_DISABLE_COPY_AND_MOVE(OvExternalMemoryHandle)

  // Caller's true size minus descriptor.offset_bytes; NOT the aligned size.
  size_t UsableBytes() const { return descriptor.size_bytes - descriptor.offset_bytes; }

  ov::Tensor ov_tensor_;  // keeps the ZeroMem pool entry alive
  void* mapped_ptr_{nullptr};

 private:
  void ReleaseSelf() { delete this; }
};

class OvExternalResourceImporter : public OrtExternalResourceImporterImpl {
 public:
  OvExternalResourceImporter(ov::Core& core, Ort::ConstEpDevice ep_device, std::string_view ov_device,
                             Ort::Logger logger);
  virtual ~OvExternalResourceImporter() = default;
  OVEP_DISABLE_COPY_AND_MOVE(OvExternalResourceImporter)

  Ort::Logger GetLogger() const { return logger_; }

 private:
  bool CanImportMemory(OrtExternalMemoryHandleType handle_type) const;
  OrtStatus* ImportMemory(const OrtExternalMemoryDescriptor* mem_desc, OrtExternalMemoryHandle** mem_handle);
  void ReleaseMemory(OrtExternalMemoryHandle* handle);  // ABI completeness only; ORT never calls it
  OrtStatus* CreateTensorFromMemory(const OrtExternalMemoryHandle* mem_handle,
                                    const OrtExternalTensorDescriptor* tensor_desc, OrtValue** value);
  void Release();

  OrtStatus* ImportD3D12Heap(const OrtExternalMemoryDescriptor& mem_desc, OrtExternalMemoryHandle** mem_handle);
  OrtStatus* ImportCpuVa(const OrtExternalMemoryDescriptor& mem_desc, OrtExternalMemoryHandle** mem_handle);

  OrtStatus* ImportNpuBuffer(const OrtExternalMemoryDescriptor& mem_desc, OrtExternalMemoryHandle** mem_handle,
                             size_t alloc_bytes, ov::intel_npu::MemType mem_type, const char* error_context);

  Ort::ConstEpDevice ep_device_;
  std::optional<ov::RemoteContext> remote_ctx_;
  Ort::Logger logger_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
