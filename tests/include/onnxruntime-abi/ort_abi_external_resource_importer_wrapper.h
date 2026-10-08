// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime_abi {
///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct external_resource_importer_wrapper {
  using device_t = typename fixture_t::device_wrapper_t;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(external_resource_importer_wrapper);
  OVEP_TESTS_DEFAULT_MOVE(external_resource_importer_wrapper);

  ///////////////////////////////////////////////////////////////////////
  external_resource_importer_wrapper(const device_t& device) {
    Ort::Status status{
        Ort::GetInteropApi().CreateExternalResourceImporterForDevice(device.ep_device_[0], &importer_)};
    if (!status.IsOK()) {
      OVEP_TESTS_THROW("Failed to create external resource importer: %s", status.GetErrorMessage().c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////
  ~external_resource_importer_wrapper(void) {
    if (importer_) {
      Ort::GetInteropApi().ReleaseExternalResourceImporter(importer_);
      importer_ = nullptr;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  bool can_import_memory(OrtExternalMemoryHandleType handle_type) const {
    bool can_import_memory = false;

    Ort::Status status{Ort::GetInteropApi().CanImportMemory(importer_, handle_type, &can_import_memory)};
    if (!status.IsOK()) {
      return false;
    }

    return can_import_memory;
  }

  ///////////////////////////////////////////////////////////////////////
  OrtExternalResourceImporter* get() const { return importer_; }

  ///////////////////////////////////////////////////////////////////////
  explicit operator bool() const { return importer_ != nullptr; }
  bool is_valid() const { return importer_ != nullptr; }

 private:
  OrtExternalResourceImporter* importer_ = nullptr;
};

}  // namespace onnxruntime_abi
}  // namespace ovep_tests
