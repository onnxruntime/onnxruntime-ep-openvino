// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
///////////////////////////////////////////////////////////////////////
struct timer_t {
  using clock_t = std::chrono::high_resolution_clock;
  using time_point_t = std::chrono::high_resolution_clock::time_point;
  using duration_t = std::chrono::duration<double>;

  ///////////////////////////////////////////////////////////////////////
  timer_t(bool implicit_start = false) {
    if (implicit_start) {
      start();
    }
  }

  ///////////////////////////////////////////////////////////////////////
  static inline auto now() { return clock_t::now(); }

  ///////////////////////////////////////////////////////////////////////
  inline void start() {
    start_ = clock_t::now();
    started_ = true;
  }

  ///////////////////////////////////////////////////////////////////////
  inline void pause() {
    if (started_ && !paused_)  // if started and not paused add current elapsed time with previously elapsed time
    {
      paused_ = true;
      elapsed_time_ = static_cast<duration_t>(clock_t::now() - start_).count() + elapsed_time_;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  inline void stop() {
    if (started_) {
      started_ = false;
      elapsed_time_ = static_cast<duration_t>(clock_t::now() - start_).count();
    } else {
      elapsed_time_ = 0;  // make it return 0.0 if not started
    }
  }

  ///////////////////////////////////////////////////////////////////////
  inline auto curr_elapsed_time_in_secs() const {
    if (started_) {
      return static_cast<duration_t>(clock_t::now() - start_).count();
    } else {
      return 0.0;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  inline bool started() const { return started_; }

  ///////////////////////////////////////////////////////////////////////
  inline auto elapsed_time_in_secs() const { return elapsed_time_; }

  ///////////////////////////////////////////////////////////////////////
  inline auto elapsed_time_in_milli_secs() const { return elapsed_time_ * 1000.0; }

  ///////////////////////////////////////////////////////////////////////
  inline auto elapsed_time_in_micro_secs() const { return elapsed_time_in_milli_secs() * 1000.0; }

 private:
  double elapsed_time_{};
  time_point_t start_;
  bool started_{false};
  bool paused_{false};
};
}  // namespace ovep_tests
