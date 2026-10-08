// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime_abi {
///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct external_memory_wrapper {
  using external_resource_importer_t = typename fixture_t::external_resource_importer_wrapper_t;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(external_memory_wrapper);

  ///////////////////////////////////////////////////////////////////////
  external_memory_wrapper(external_resource_importer_t& importer, const OrtExternalMemoryDescriptor& mem_desc)
      : importer_(importer), handle_(nullptr) {
    Ort::Status status{Ort::GetInteropApi().ImportMemory(importer_.get(), &mem_desc, &handle_)};
    if (!status.IsOK()) {
      OVEP_TESTS_THROW("Failed to import external memory: %s", status.GetErrorMessage().c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////
  external_memory_wrapper(
      external_resource_importer_t& importer,
      void* native_handle,
      size_t alloc_size,
      OrtExternalMemoryHandleType handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
      size_t offset_bytes = 0)
      : importer_(importer), handle_(nullptr) {
    OrtExternalMemoryDescriptor mem_desc{
        .version = ORT_API_VERSION,
        .handle_type = handle_type,
        .native_handle = native_handle,
        .size_bytes = alloc_size,
        .offset_bytes = offset_bytes,
    };

    Ort::Status status{Ort::GetInteropApi().ImportMemory(importer_.get(), &mem_desc, &handle_)};
    if (!status.IsOK()) {
      OVEP_TESTS_THROW("Failed to import external memory: %s", status.GetErrorMessage().c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////
  ~external_memory_wrapper() {
    if (handle_) {
      Ort::GetInteropApi().ReleaseExternalMemoryHandle(handle_);
      handle_ = nullptr;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  OrtValue* create_tensor(const OrtExternalTensorDescriptor& desc) const {
    OrtValue* value = nullptr;

    Ort::Status status{Ort::GetInteropApi().CreateTensorFromMemory(importer_.get(), handle_, &desc, &value)};
    if (!status.IsOK()) {
      OVEP_TESTS_THROW("Failed to create tensor from memory: %s", status.GetErrorMessage().c_str());
    }

    return value;
  }

  ///////////////////////////////////////////////////////////////////////
  OrtValue* create_tensor(
      const std::vector<int64_t>& dims,
      ONNXTensorElementDataType element_type,
      size_t offset_bytes = 0) const {
    OrtExternalTensorDescriptor desc{
        .version = ORT_API_VERSION,
        .element_type = element_type,
        .shape = dims.data(),
        .rank = dims.size(),
        .offset_bytes = offset_bytes,
    };

    OrtValue* value = nullptr;

    Ort::Status status{Ort::GetInteropApi().CreateTensorFromMemory(importer_.get(), handle_, &desc, &value)};
    if (status) {
      OVEP_TESTS_THROW("Failed to create tensor from memory: %s", status.GetErrorMessage().c_str());
    }

    return value;
  }

  ///////////////////////////////////////////////////////////////////////
  OrtExternalMemoryHandle* get() const { return handle_; }

 private:
  external_resource_importer_t& importer_;
  OrtExternalMemoryHandle* handle_ = nullptr;
};

}  // namespace onnxruntime_abi
}  // namespace ovep_tests
