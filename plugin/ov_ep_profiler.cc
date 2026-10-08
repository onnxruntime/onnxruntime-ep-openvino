// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ov_ep_profiler.h"

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <utility>

namespace onnxruntime {
namespace openvino_ep_plugin {

namespace {

constexpr int32_t kTidBlockSize = 16;
constexpr int32_t kOverflowBaseTid = 1024;
constexpr int32_t kOverflowTypeStride = 256;

size_t GetExecTypeIndex(std::string_view exec_type) {
  static constexpr std::array<std::string_view, 4> kExecTypeOrder = {
      "DMA",
      "DPU",
      "Shave",
      "Other",
  };

  for (size_t index = 0; index < kExecTypeOrder.size(); ++index) {
    if (exec_type == kExecTypeOrder[index]) {
      return index;
    }
  }

  return kExecTypeOrder.size() - 1;
}

int32_t GetExecTypeBaseTid(std::string_view exec_type) {
  return static_cast<int32_t>(GetExecTypeIndex(exec_type)) * kTidBlockSize;
}

std::string_view NormalizeExecType(std::string_view exec_type) {
  if (exec_type == "DMA" || exec_type == "DPU" || exec_type == "Shave") {
    return exec_type;
  }

  return "Other";
}

int64_t ComputeIntervalEndUs(int64_t start_us, int64_t duration_us) {
  if (duration_us <= 0) {
    return start_us;
  }

  if (start_us > std::numeric_limits<int64_t>::max() - duration_us) {
    return std::numeric_limits<int64_t>::max();
  }

  return start_us + duration_us;
}

class OverlapAwareTidAllocator {
 public:
  int32_t Allocate(std::string_view exec_type, int64_t start_us, int64_t duration_us) {
    const std::string normalized_exec_type{NormalizeExecType(exec_type)};
    const size_t exec_type_index = GetExecTypeIndex(normalized_exec_type);
    auto& lane_end_times = lane_end_times_by_exec_type_[normalized_exec_type];
    const int64_t end_us = ComputeIntervalEndUs(start_us, duration_us);

    for (size_t lane_index = 0; lane_index < lane_end_times.size(); ++lane_index) {
      if (start_us >= lane_end_times[lane_index]) {
        lane_end_times[lane_index] = end_us;
        if (lane_index < static_cast<size_t>(kTidBlockSize)) {
          return GetExecTypeBaseTid(normalized_exec_type) + static_cast<int32_t>(lane_index);
        }

        return kOverflowBaseTid + static_cast<int32_t>(exec_type_index) * kOverflowTypeStride +
               static_cast<int32_t>(lane_index - kTidBlockSize);
      }
    }

    lane_end_times.push_back(end_us);
    const size_t lane_index = lane_end_times.size() - 1;
    if (lane_index < static_cast<size_t>(kTidBlockSize)) {
      return GetExecTypeBaseTid(normalized_exec_type) + static_cast<int32_t>(lane_index);
    }

    return kOverflowBaseTid + static_cast<int32_t>(exec_type_index) * kOverflowTypeStride +
           static_cast<int32_t>(lane_index - kTidBlockSize);
  }

 private:
  std::unordered_map<std::string, std::vector<int64_t>> lane_end_times_by_exec_type_;
};

}  // namespace

thread_local std::vector<OVEpProfilerImpl::TlsEventFrame> OVEpProfilerImpl::tls_event_stack_{};

OVEpProfilerImpl::OVEpProfilerImpl(Ort::Logger logger)
    : OrtEpProfilerImpl{},
      logger_(std::move(logger)) {
  ort_version_supported = ort_runtime_api_version;

  OrtEpProfilerImpl::Release = OVEP_API_ENTRY_TEARDOWN(OVEpProfilerImpl, Release);
  OrtEpProfilerImpl::StartProfiling = OVEP_API_ENTRY(OVEpProfilerImpl, StartProfiling);
  OrtEpProfilerImpl::StartEvent = OVEP_API_ENTRY(OVEpProfilerImpl, StartEvent);
  OrtEpProfilerImpl::StopEvent = OVEP_API_ENTRY(OVEpProfilerImpl, StopEvent);
  OrtEpProfilerImpl::EndProfiling = OVEP_API_ENTRY(OVEpProfilerImpl, EndProfiling);
}

void OVEpProfilerImpl::Release() {
  delete this;
}

OrtStatus* OVEpProfilerImpl::StartProfiling(int64_t ep_profiling_start_offset_ns) {
  {
    std::scoped_lock<std::mutex> lock(cache_mutex_);
    ep_profiling_start_offset_ns_ = ep_profiling_start_offset_ns;
    ep_profiling_start_time_ = std::chrono::steady_clock::now();
    profiling_started_ = true;
    ort_events_.clear();
    pending_ov_events_.clear();
  }

  // Notify EP that profiling has started. Call outside the lock to avoid potential deadlock.
  if (profiling_state_callback_) {
    profiling_state_callback_(true);
  }

  return nullptr;
}

OrtStatus* OVEpProfilerImpl::StartEvent(uint64_t ort_event_correlation_id) {
  tls_event_stack_.push_back(TlsEventFrame{this, ort_event_correlation_id});
  return nullptr;
}

OrtStatus* OVEpProfilerImpl::StopEvent(uint64_t ort_event_correlation_id,
                                       const OrtProfilingEvent* ort_event) {
  if (!ort_event) {
    return nullptr;
  }

  ParentOrtEvent snapshot{};
  snapshot.parent_correlation_id = PopTlsEventAndGetParent(ort_event_correlation_id);

  const Ort::ConstProfilingEvent event{ort_event};
  snapshot.category = event.GetCategory();

  const char* event_name = event.GetName();
  if (event_name) {
    snapshot.name = event_name;
  }

  snapshot.timestamp_us = event.GetTimestampUs();
  snapshot.duration_us = event.GetDurationUs();

  std::scoped_lock<std::mutex> lock(cache_mutex_);
  ort_events_[ort_event_correlation_id] = std::move(snapshot);
  return nullptr;
}

OrtStatus* OVEpProfilerImpl::EndProfiling(OrtProfilingEventsContainer* events_container) {
  if (!events_container) {
    return nullptr;
  }

  std::unordered_map<uint64_t, ParentOrtEvent> ort_events;
  std::vector<PendingOvEvent> pending_events;
  {
    std::scoped_lock<std::mutex> lock(cache_mutex_);
    ort_events = std::move(ort_events_);
    pending_events = std::move(pending_ov_events_);
    ort_events_.clear();
    pending_ov_events_.clear();
    profiling_started_ = false;
  }

  // Notify EP that profiling has ended. Call after all profiling data is processed.
  if (profiling_state_callback_) {
    profiling_state_callback_(false);
  }

  struct ParentIntervalStats {
    int64_t min_start_us{std::numeric_limits<int64_t>::max()};
  };

  std::unordered_map<uint64_t, ParentIntervalStats> parent_interval_stats;
  for (const auto& pending : pending_events) {
    auto& stats = parent_interval_stats[pending.ort_event_correlation_id];
    stats.min_start_us = std::min(stats.min_start_us, pending.start_time_us);
  }
  std::stable_sort(pending_events.begin(), pending_events.end(),
                   [&ort_events](const PendingOvEvent& left, const PendingOvEvent& right) {
                     auto resolve_parent_timestamp = [&ort_events](const PendingOvEvent& event) {
                       const auto parent_it = ort_events.find(event.ort_event_correlation_id);
                       return parent_it == ort_events.end() ? int64_t{0} : parent_it->second.timestamp_us;
                     };

                     if (left.ort_event_correlation_id != right.ort_event_correlation_id) {
                       return resolve_parent_timestamp(left) < resolve_parent_timestamp(right);
                     }

                     return left.start_time_us < right.start_time_us;
                   });

  OverlapAwareTidAllocator tid_allocator;

  for (const auto& pending : pending_events) {
    const auto parent_it = ort_events.find(pending.ort_event_correlation_id);
    const ParentOrtEvent* parent = parent_it == ort_events.end() ? nullptr : &parent_it->second;

    int64_t timestamp_us = 0;
    int64_t duration_us = std::max<int64_t>(0, pending.duration_us);

    if (parent) {
      const int64_t parent_begin_us = parent->timestamp_us;
      const auto stats_it = parent_interval_stats.find(pending.ort_event_correlation_id);
      if (stats_it != parent_interval_stats.end()) {
        const auto& stats = stats_it->second;
        timestamp_us = parent_begin_us + std::max<int64_t>(0, pending.start_time_us - stats.min_start_us);
      } else {
        timestamp_us = parent_begin_us;
      }
    }

    const int32_t tid = tid_allocator.Allocate(pending.exec_type, timestamp_us, duration_us);

    OVEP_RETURN_IF_ERROR(AddOvEventToContainer(pending, parent, timestamp_us, duration_us, tid, events_container));
  }

  return nullptr;
}

void OVEpProfilerImpl::CaptureInferRequestProfiling(const ov::InferRequest& infer_request) {
  const TlsEventFrame* frame = GetCurrentTlsFrame();
  if (!frame || !frame->profiler) {
    return;
  }

  // Only collect OV events while ORT profiling is active.
  {
    std::scoped_lock<std::mutex> lock(frame->profiler->cache_mutex_);
    if (!frame->profiler->profiling_started_) {
      return;
    }
  }

  try {
    const auto profiling_info = infer_request.get_profiling_info();
    if (profiling_info.empty()) {
      return;
    }

    std::vector<PendingOvEvent> pending_events;
    pending_events.reserve(profiling_info.size());

    for (const auto& info : profiling_info) {
      const int64_t duration_us = info.real_time.count();
      if (info.status != ov::ProfilingInfo::Status::EXECUTED || duration_us <= 0) {
        continue;
      }

      PendingOvEvent pending{};
      pending.ort_event_correlation_id = frame->correlation_id;
      pending.duration_us = duration_us;
      pending.start_time_us = info.start_time.count();

      // Metadata copy is best-effort; avoid propagating allocation failures.
      try {
        pending.node_name = info.node_name;
        pending.node_type = info.node_type;
        pending.exec_type = info.exec_type;
      } catch (const std::bad_alloc&) {
        ORT_CXX_LOGF(frame->profiler->logger_,
                     ORT_LOGGING_LEVEL_WARNING,
                     "Failed to copy OV profiling metadata due to allocation failure. correlation_id=%llu duration_us=%lld",
                     static_cast<unsigned long long>(frame->correlation_id),
                     static_cast<long long>(duration_us));
      }

      pending.status = ToStatusString(info.status);
      pending_events.push_back(std::move(pending));
    }

    if (pending_events.empty()) {
      return;
    }

    std::scoped_lock<std::mutex> lock(frame->profiler->cache_mutex_);
    auto& queue = frame->profiler->pending_ov_events_;
    queue.insert(queue.end(), std::make_move_iterator(pending_events.begin()), std::make_move_iterator(pending_events.end()));
  } catch (const std::exception& e) {
    ORT_CXX_LOGF(frame->profiler->logger_,
                 ORT_LOGGING_LEVEL_WARNING,
                 "CaptureInferRequestProfiling swallowed exception: %s",
                 e.what());
    return;
  } catch (...) {
    ORT_CXX_LOG(frame->profiler->logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "CaptureInferRequestProfiling swallowed non-std exception.");
    // Profiling should never affect inference stability.
    return;
  }
}

std::optional<uint64_t> OVEpProfilerImpl::PopTlsEventAndGetParent(uint64_t ort_event_correlation_id) {
  auto current_it = std::find_if(tls_event_stack_.rbegin(), tls_event_stack_.rend(),
                                 [this](const TlsEventFrame& frame) {
                                   return frame.profiler == this;
                                 });

  if (current_it == tls_event_stack_.rend()) {
    return std::nullopt;
  }

  const auto current_index = static_cast<size_t>(std::distance(current_it, tls_event_stack_.rend()) - 1);
  std::optional<uint64_t> parent_id;

  if (tls_event_stack_[current_index].correlation_id == ort_event_correlation_id) {
    for (size_t i = current_index; i-- > 0;) {
      if (tls_event_stack_[i].profiler == this) {
        parent_id = tls_event_stack_[i].correlation_id;
        break;
      }
    }

    tls_event_stack_.erase(tls_event_stack_.begin() + static_cast<std::ptrdiff_t>(current_index));
    return parent_id;
  }

  auto matched_it = std::find_if(tls_event_stack_.rbegin(), tls_event_stack_.rend(),
                                 [this, ort_event_correlation_id](const TlsEventFrame& frame) {
                                   return frame.profiler == this && frame.correlation_id == ort_event_correlation_id;
                                 });

  if (matched_it == tls_event_stack_.rend()) {
    return std::nullopt;
  }

  const auto matched_index = static_cast<size_t>(std::distance(matched_it, tls_event_stack_.rend()) - 1);
  for (size_t i = matched_index; i-- > 0;) {
    if (tls_event_stack_[i].profiler == this) {
      parent_id = tls_event_stack_[i].correlation_id;
      break;
    }
  }

  tls_event_stack_.erase(tls_event_stack_.begin() + static_cast<std::ptrdiff_t>(matched_index));
  return parent_id;
}

const OVEpProfilerImpl::TlsEventFrame* OVEpProfilerImpl::GetCurrentTlsFrame() {
  for (auto it = tls_event_stack_.rbegin(); it != tls_event_stack_.rend(); ++it) {
    if (it->profiler != nullptr) {
      return &(*it);
    }
  }

  return nullptr;
}

std::string OVEpProfilerImpl::ToStatusString(ov::ProfilingInfo::Status status) {
  switch (status) {
    case ov::ProfilingInfo::Status::EXECUTED:
      return "EXECUTED";
    case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
      return "OPTIMIZED_OUT";
    case ov::ProfilingInfo::Status::NOT_RUN:
      return "NOT_RUN";
    default:
      return "UNKNOWN";
  }
}

std::string OVEpProfilerImpl::BuildEventName(const PendingOvEvent& event) {
  if (!event.node_name.empty()) {
    return std::string{"OV::"} + event.node_name;
  }

  if (!event.node_type.empty()) {
    return std::string{"OV::"} + event.node_type;
  }

  return "OV::UnknownNode";
}

OrtStatus* OVEpProfilerImpl::AddOvEventToContainer(const PendingOvEvent& pending,
                                                   const ParentOrtEvent* parent,
                                                   int64_t timestamp_us,
                                                   int64_t duration_us,
                                                   int32_t tid,
                                                   OrtProfilingEventsContainer* events_container) const {
  const auto event_name = BuildEventName(pending);
  const auto correlation_id = std::to_string(pending.ort_event_correlation_id);
  const std::string parent_name = parent ? parent->name : std::string{};

  const char* arg_keys[] = {
      "ep",
      "ort_correlation_id",
      "parent_ort_event_name",
      "ov_status",
      "ov_node_type",
      "ov_exec_type"};
  const char* arg_values[] = {
      "OpenVINOExecutionProvider",
      correlation_id.c_str(),
      parent_name.c_str(),
      pending.status.c_str(),
      pending.node_type.c_str(),
      pending.exec_type.c_str()};

  Ort::ProfilingEvent profiling_event{OrtProfilingEventCategory_KERNEL,
                                      -1,
                                      tid,
                                      event_name.c_str(),
                                      timestamp_us,
                                      duration_us,
                                      arg_keys,
                                      arg_values,
                                      std::size(arg_keys)};

  const OrtProfilingEvent* events[] = {profiling_event};
  Ort::UnownedProfilingEventsContainer container{events_container};
  OVEP_RETURN_IF_ERROR(container.AddEvents(events, std::size(events)).release());

  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
