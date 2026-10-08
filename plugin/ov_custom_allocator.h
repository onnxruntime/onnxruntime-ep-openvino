// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <mutex>
#include <string_view>
#include "openvino/runtime/core.hpp"

#include "ov_utils.h"
#include "ov_device.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class BaseAllocator : public OrtAllocator {
 public:
  BaseAllocator() { version = ORT_API_VERSION; }
  virtual ~BaseAllocator() = default;
};

class OvBaseAllocator : public BaseAllocator, public OvLoggable {
 public:
  OvBaseAllocator(Ort::Logger logger) : BaseAllocator(), OvLoggable(logger) {}
  virtual ~OvBaseAllocator() = default;

  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const OrtMemoryInfo* Info() const = 0;
};

class OVRTAllocator : public OvBaseAllocator {
 public:
  static constexpr std::string_view allocator_prefix{"OpenVINO_shared_"};
  static constexpr size_t alignment_ = 4096;

  OVRTAllocator(ov::Core& core, const OrtMemoryInfo& mem_info, const OvDevice& ov_device, Ort::Logger logger);
  virtual ~OVRTAllocator() = default;

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo* Info() const override;

 private:
  ov::Core& core_;
  std::optional<ov::RemoteContext> remote_ctx_;  // Only for NPU/GPU
  std::unordered_map<void*, ov::Tensor*> allocated_;
  const OrtMemoryInfo* memory_info_;
  std::mutex mutex_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
