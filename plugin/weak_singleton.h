// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>

namespace onnxruntime {
namespace openvino_ep_plugin {

template <typename T>
class WeakSingleton {
 public:
  static std::shared_ptr<T> Get() {
    static std::weak_ptr<T> instance;
    static std::mutex mutex;

    // Acquire the mutex before any access to `instance`. `std::weak_ptr` itself
    // is not thread-safe: an unsynchronized `instance.lock()` racing with the
    // `instance = ptr` write below is a data race. Holding the lock across
    // both the read and the write serializes all access to `instance`.
    std::lock_guard<std::mutex> lock(mutex);
    auto ptr = instance.lock();
    if (!ptr) {
      ptr = std::make_shared<T>();
      instance = ptr;
    }
    return ptr;
  }

 protected:
  WeakSingleton() = default;
  virtual ~WeakSingleton() = default;
  WeakSingleton(const WeakSingleton&) = delete;
  WeakSingleton(WeakSingleton&&) = delete;
  WeakSingleton& operator=(const WeakSingleton&) = delete;
  WeakSingleton& operator=(WeakSingleton&&) = delete;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
