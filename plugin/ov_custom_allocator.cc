// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#else
#include <cstdlib>  // For posix_memalign and free
#endif

#include "ov_custom_allocator.h"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

static void* AlignedAlloc(size_t size, size_t alignment) {
  if (size == 0) return nullptr;

  void* ptr;
#if _MSC_VER
  ptr = _aligned_malloc(size, alignment);
  if (ptr == nullptr)
    throw std::bad_alloc();
#else
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0)
    throw std::bad_alloc();
#endif
  return ptr;
}

static void AlignedFree(void* ptr) {
  if (ptr == nullptr) return;

#ifdef _WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

OVRTAllocator::OVRTAllocator(ov::Core& core, const OrtMemoryInfo& mem_info, const OvDevice& ov_device, Ort::Logger logger)
    : OvBaseAllocator{logger},
      core_(core),
      memory_info_(&mem_info) {
  try {
    remote_ctx_ = core_.get_default_context(ov_device);
  } catch (...) {
    // Fallback to CPU allocator
  }

  OrtAllocator::Alloc = OVEP_API_ENTRY(OVRTAllocator, Alloc);
  OrtAllocator::Free = OVEP_API_ENTRY(OVRTAllocator, Free);
  OrtAllocator::Info = OVEP_API_ENTRY(OVRTAllocator, Info);
}

void* OVRTAllocator::Alloc(size_t size) {
  if (!remote_ctx_) {
    return AlignedAlloc(size, alignment_);
  } else {
    // Use OpenVINO remote tensor allocation for non-CPU devices
    ov::Tensor* tensor = new ov::Tensor(
        remote_ctx_.value().create_host_tensor(ov::element::Type_t::u8, {size}));
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_.insert({tensor->data(), tensor});
    return tensor->data();
  }
}

void OVRTAllocator::Free(void* p) {
  if (!remote_ctx_) {
    AlignedFree(p);
  } else {
    // For non-CPU allocation, find and delete the tensor
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_.find(p);
    if (it != allocated_.end()) {
      delete it->second;
      allocated_.erase(it);
    }
  }
}

const OrtMemoryInfo* OVRTAllocator::Info() const {
  return memory_info_;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
