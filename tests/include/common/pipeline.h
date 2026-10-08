// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace common {
///////////////////////////////////////////////////////////////////////
struct pipeline_t {
 protected:
  std::condition_variable cond_var_;
  std::mutex mutex_;
  bool available_ = true;

 public:
  ///////////////////////////////////////////////////////////////////////
  virtual void start_async(void) = 0;
  virtual void wait_for_completion(void) = 0;

  ///////////////////////////////////////////////////////////////////////
  virtual void run(void) {
    start_async();

    wait_for_completion();
  }

  ///////////////////////////////////////////////////////////////////////////
  void acquire(void) {
    // Wait for pipeline to be available
    std::unique_lock<std::mutex> lock(mutex_);

    cond_var_.wait(lock, [&] { return available_; });
    available_ = false;
  }

  ///////////////////////////////////////////////////////////////////////////
  void release(void) {
    // Signal that pipeline is available
    std::unique_lock<std::mutex> lock(mutex_);

    available_ = true;
    cond_var_.notify_one();
  }

  ///////////////////////////////////////////////////////////////////////////
  virtual ~pipeline_t(void) {}
};
}  // namespace common
}  // namespace ovep_tests
