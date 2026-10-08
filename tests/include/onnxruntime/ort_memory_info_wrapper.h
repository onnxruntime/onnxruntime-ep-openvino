// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime {
namespace detail {
///////////////////////////////////////////////////////////////////////
// Owns the CPU fallback memory info (null on the OpenVINO path). A base so it
// is constructed before the ConstMemoryInfo view that borrows from it.
struct cpu_memory_info_owner {
  Ort::MemoryInfo owned_{nullptr};
};
}  // namespace detail

///////////////////////////////////////////////////////////////////////
// Non-owning view of a device's memory info: the EP's registered OrtMemoryInfo
// for OpenVINO devices, or an owned CPU memory info for non-OV (MLAS) devices.
// Using the registered handle directly keeps it equal to the allocator's, so
// allocator lookups match.
template <typename fixture_t>
struct memory_info_wrapper : private detail::cpu_memory_info_owner, Ort::ConstMemoryInfo {
  using device_t = typename fixture_t::device_wrapper_t;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(memory_info_wrapper);

  ///////////////////////////////////////////////////////////////////////
  memory_info_wrapper(const device_t& device, OrtDeviceMemoryType mem_type)
      : detail::cpu_memory_info_owner{make_owned(device, mem_type)},
        Ort::ConstMemoryInfo{select_info(device, mem_type, owned_)} {}

  ///////////////////////////////////////////////////////////////////////
  // Borrows a caller-provided memory info. Owns nothing; the caller must keep
  // the underlying OrtMemoryInfo alive for this wrapper's lifetime.
  explicit memory_info_wrapper(Ort::ConstMemoryInfo borrowed) : Ort::ConstMemoryInfo{borrowed} {}

 private:
  ///////////////////////////////////////////////////////////////////////
  // The EP device's registered memory info, or null when there is none to use.
  static Ort::ConstMemoryInfo ep_device_memory_info(const device_t& device, OrtDeviceMemoryType mem_type) {
    if (device.ort_execution_provider_ != ort_execution_provider_t::ov || device.ep_device_.empty()) {
      return Ort::ConstMemoryInfo{nullptr};
    }
    return device.ep_device_.front().GetMemoryInfo(mem_type);
  }

  ///////////////////////////////////////////////////////////////////////
  // Owns a CPU memory info only when there is no EP-device one to borrow.
  static Ort::MemoryInfo make_owned(const device_t& device, OrtDeviceMemoryType mem_type) {
    if (ep_device_memory_info(device, mem_type)) {
      return Ort::MemoryInfo{nullptr};
    }
    return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  }

  ///////////////////////////////////////////////////////////////////////
  static Ort::ConstMemoryInfo select_info(const device_t& device, OrtDeviceMemoryType mem_type,
                                          const Ort::MemoryInfo& owned) {
    if (auto ep_mi = ep_device_memory_info(device, mem_type)) {
      return ep_mi;
    }
    return owned.GetConst();
  }
};

}  // namespace onnxruntime
}  // namespace ovep_tests
