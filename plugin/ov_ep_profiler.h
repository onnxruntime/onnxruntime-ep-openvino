// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/profiling_info.hpp"

#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class OVEpProfilerImpl : public OrtEpProfilerImpl {
 public:
  explicit OVEpProfilerImpl(Ort::Logger logger);
  ~OVEpProfilerImpl() = default;

  OVEP_DISABLE_COPY_AND_MOVE(OVEpProfilerImpl)

  // Callback to notify EP about profiling state changes. Allows EP to synchronize
  // its own profiling flags with the profiler's actual state.
  using ProfilingStateCallback = std::function<void(bool profiling_enabled)>;
  void SetProfilingStateCallback(ProfilingStateCallback callback) {
    profiling_state_callback_ = std::move(callback);
  }

  Ort::Logger GetLogger() const noexcept {
    return logger_;
  }

  void Release();
  OrtStatus* StartProfiling(int64_t ep_profiling_start_offset_ns);
  OrtStatus* StartEvent(uint64_t ort_event_correlation_id);
  OrtStatus* StopEvent(uint64_t ort_event_correlation_id, const OrtProfilingEvent* ort_event);
  OrtStatus* EndProfiling(OrtProfilingEventsContainer* events_container);

  // Called from compute path after ov::InferRequest::infer().
  static void CaptureInferRequestProfiling(const ov::InferRequest& infer_request);

 private:
  struct ParentOrtEvent {
    std::string name;
    OrtProfilingEventCategory category{OrtProfilingEventCategory_SESSION};
    int64_t timestamp_us{0};
    int64_t duration_us{0};
    std::optional<uint64_t> parent_correlation_id;
  };

  struct TlsEventFrame {
    OVEpProfilerImpl* profiler{nullptr};
    uint64_t correlation_id{0};
  };

  struct PendingOvEvent {
    uint64_t ort_event_correlation_id{0};
    int64_t duration_us{0};
    int64_t start_time_us{0};
    std::string node_name;
    std::string node_type;
    std::string exec_type;
    std::string status;
  };
  std::optional<uint64_t> PopTlsEventAndGetParent(uint64_t ort_event_correlation_id);
  static const TlsEventFrame* GetCurrentTlsFrame();
  static std::string ToStatusString(ov::ProfilingInfo::Status status);
  static std::string BuildEventName(const PendingOvEvent& event);
  OrtStatus* AddOvEventToContainer(const PendingOvEvent& pending,
                                   const ParentOrtEvent* parent,
                                   int64_t timestamp_us,
                                   int64_t duration_us,
                                   int32_t tid,
                                   OrtProfilingEventsContainer* events_container) const;

 private:
  static thread_local std::vector<TlsEventFrame> tls_event_stack_;

  const OrtEpApi& ep_api_{Ort::GetEpApi()};
  Ort::Logger logger_;

  int64_t ep_profiling_start_offset_ns_{0};
  std::chrono::steady_clock::time_point ep_profiling_start_time_{};
  bool profiling_started_{false};

  ProfilingStateCallback profiling_state_callback_;  // Callback to sync profiling state with EP

  std::mutex cache_mutex_;
  std::unordered_map<uint64_t, ParentOrtEvent> ort_events_;
  std::vector<PendingOvEvent> pending_ov_events_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
