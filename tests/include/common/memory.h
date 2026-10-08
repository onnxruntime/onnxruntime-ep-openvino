// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <memory>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <optional>

///////////////////////////////////////////////////////////////////////
// Utility functions
namespace memory_utils {

inline constexpr double byte_to_mb(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline constexpr double kb_to_mb(size_t kb) {
  return static_cast<double>(kb) / 1024.0;
}

inline std::string to_string(const LUID& luid) {
  std::stringstream stream;
  stream << "0x" << std::setw(8) << std::setfill('0') << std::hex << luid.HighPart;
  stream << "_";
  stream << "0x" << std::setw(8) << std::setfill('0') << std::hex << luid.LowPart;
  return stream.str();
}

}  // namespace memory_utils

///////////////////////////////////////////////////////////////////////
struct cpu_mem_counter_snapshot_t {
  double virtual_memory_ = 0;          // Total virtual memory (page file on Windows)
  double resident_memory_ = 0;         // Memory in RAM (working set on Windows)
  double resident_shared_memory_ = 0;  // Shared memory in RAM (file-backed + shared libs)

  cpu_mem_counter_snapshot_t() = default;
  cpu_mem_counter_snapshot_t operator-(const cpu_mem_counter_snapshot_t& other) const {
    cpu_mem_counter_snapshot_t delta;
    delta.virtual_memory_ = virtual_memory_ - other.virtual_memory_;
    delta.resident_memory_ = resident_memory_ - other.resident_memory_;
    delta.resident_shared_memory_ = resident_shared_memory_ - other.resident_shared_memory_;
    return delta;
  }

  friend std::ostream& operator<<(std::ostream& os, const cpu_mem_counter_snapshot_t& snapshot) {
    os << "CPU Memory Snapshot:\n"
       << "  Virtual:  " << std::fixed << std::setprecision(2) << snapshot.virtual_memory_ << " MB\n"
       << "  Resident: " << std::fixed << std::setprecision(2) << snapshot.resident_memory_ << " MB\n"
       << "  Shared:   " << std::fixed << std::setprecision(2) << snapshot.resident_shared_memory_ << " MB";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////
struct cpu_mem_counter_t {
  HANDLE process_{};
  cpu_mem_counter_snapshot_t start_;

  ///////////////////////////////////////////////////////////////////////
  cpu_mem_counter_t()
      : process_(OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, GetCurrentProcessId())) {
    if (!process_) {
      OVEP_TESTS_THROW("Failed to open current process for memory monitoring");
    }
  }

  ///////////////////////////////////////////////////////////////////////
  virtual ~cpu_mem_counter_t() {
    if (process_) {
      CloseHandle(process_);
    }
  }

  // Delete copy/move to prevent resource issues
  cpu_mem_counter_t(const cpu_mem_counter_t&) = delete;
  cpu_mem_counter_t& operator=(const cpu_mem_counter_t&) = delete;
  cpu_mem_counter_t(cpu_mem_counter_t&&) = delete;
  cpu_mem_counter_t& operator=(cpu_mem_counter_t&&) = delete;

  cpu_mem_counter_snapshot_t snapshot() const { return get_snapshot(); }
  void start() { start_ = snapshot(); }
  cpu_mem_counter_snapshot_t delta_from_start() const { return snapshot() - start_; }

 private:
  ///////////////////////////////////////////////////////////////////////
  size_t get_shared_working_set_size() const {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(process_, &pmc, sizeof(pmc))) {
      return 0;
    }

    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    size_t page_size = si.dwPageSize;

    // Estimate number of pages.
    size_t estimated_pages = (pmc.WorkingSetSize / page_size);

    // Loop to handle potential buffer resizing (if race condition where WS grows)
    for (int attempt = 0; attempt < 2; ++attempt) {
      // Add margin for growth between GetProcessMemoryInfo and QueryWorkingSet
      size_t buffer_size = sizeof(PSAPI_WORKING_SET_INFORMATION) +
                           (estimated_pages + 5000) * sizeof(PSAPI_WORKING_SET_BLOCK);

      auto buffer = std::make_unique<char[]>(buffer_size);
      auto* wsi = reinterpret_cast<PSAPI_WORKING_SET_INFORMATION*>(buffer.get());

      if (QueryWorkingSet(process_, wsi, static_cast<DWORD>(buffer_size))) {
        size_t shared_bytes = 0;
        for (ULONG_PTR i = 0; i < wsi->NumberOfEntries; ++i) {
          if (wsi->WorkingSetInfo[i].Shared) {
            shared_bytes += page_size;
          }
        }
        return shared_bytes;
      }

      if (GetLastError() != ERROR_BAD_LENGTH) {
        break;  // Some other error
      }

      // If bad length, increase estimate significantly for next try
      estimated_pages *= 2;
    }

    return 0;
  }

  ///////////////////////////////////////////////////////////////////////
  cpu_mem_counter_snapshot_t get_snapshot() const {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(process_, &pmc, sizeof(pmc))) {
      return cpu_mem_counter_snapshot_t{};
    }

    cpu_mem_counter_snapshot_t snapshot;
    snapshot.virtual_memory_ = memory_utils::byte_to_mb(pmc.PagefileUsage);
    snapshot.resident_memory_ = memory_utils::byte_to_mb(pmc.WorkingSetSize);

    // Use accurate shared memory calculation
    size_t shared_bytes = get_shared_working_set_size();
    snapshot.resident_shared_memory_ = memory_utils::byte_to_mb(shared_bytes);

    return snapshot;
  }
};

///////////////////////////////////////////////////////////////////////
struct gpu_perf_query_t {
  HQUERY query_{};

  ///////////////////////////////////////////////////////////////////////
  gpu_perf_query_t() {
    PDH_STATUS status = PdhOpenQuery(nullptr, 0, &query_);
    if (status != ERROR_SUCCESS) {
      OVEP_TESTS_THROW("Failed to open PDH query (status: %d)", status);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  virtual ~gpu_perf_query_t() {
    if (query_) {
      PdhCloseQuery(query_);
      query_ = nullptr;
    }
  }

  // Delete copy/move
  gpu_perf_query_t(const gpu_perf_query_t&) = delete;
  gpu_perf_query_t& operator=(const gpu_perf_query_t&) = delete;
  gpu_perf_query_t(gpu_perf_query_t&&) = delete;
  gpu_perf_query_t& operator=(gpu_perf_query_t&&) = delete;

  ///////////////////////////////////////////////////////////////////////
  void add_counter(const std::string& path, HCOUNTER& counter) {
    PDH_STATUS status = PdhAddCounterA(query_, path.c_str(), 0, &counter);
    if (status != ERROR_SUCCESS) {
      OVEP_TESTS_THROW(
          "Failed to add PDH counter '%s' (status: %d)", path.c_str(), status);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  bool run() {
    PDH_STATUS status = PdhCollectQueryData(query_);
    return (status == ERROR_SUCCESS);
  }
};

///////////////////////////////////////////////////////////////////////
struct gpu_mem_snapshot_t {
  double shared_mem_usage_ = 0.0;

  gpu_mem_snapshot_t() = default;
  gpu_mem_snapshot_t(double shared_mem_usage) : shared_mem_usage_(shared_mem_usage) {}

  gpu_mem_snapshot_t operator-(const gpu_mem_snapshot_t& other) const {
    gpu_mem_snapshot_t delta;
    delta.shared_mem_usage_ = shared_mem_usage_ - other.shared_mem_usage_;
    return delta;
  }

  friend std::ostream& operator<<(std::ostream& os, const gpu_mem_snapshot_t& snapshot) {
    os << "GPU Memory Snapshot:\n"
       << "  Shared:  " << std::fixed << std::setprecision(2) << snapshot.shared_mem_usage_ << " MB\n";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////
struct gpu_mem_counter_t {
  const LUID& luid_;
  gpu_mem_snapshot_t start_;
  std::unique_ptr<gpu_perf_query_t> query_;
  HCOUNTER gpu_shared_mem_usage_counter_{};

  ///////////////////////////////////////////////////////////////////////
  gpu_mem_counter_t(const LUID& luid) : luid_(luid) {
    std::stringstream gpu_shared_mem_usage_counter_stream;
    gpu_shared_mem_usage_counter_stream << "\\GPU Process Memory(pid_" << std::to_string(GetCurrentProcessId())
                                        << "_luid_" << memory_utils::to_string(luid_) << "_phys_0)\\Shared Usage";

    query_ = std::make_unique<gpu_perf_query_t>();  // Throws if PDH query fails
    query_->add_counter(
        gpu_shared_mem_usage_counter_stream.str(),
        gpu_shared_mem_usage_counter_);  // Throws if counter fails
  }

  ///////////////////////////////////////////////////////////////////////
  gpu_mem_snapshot_t snapshot() const { return gpu_mem_snapshot_t(get_gpu_shared_mem_usage()); }
  void start() { start_ = snapshot(); }
  gpu_mem_snapshot_t delta_from_start() const { return snapshot() - start_; }

 private:
  ///////////////////////////////////////////////////////////////////////
  double get_gpu_shared_mem_usage() const {
    if (!query_->run()) {
      return 0.0;
    }

    return memory_utils::byte_to_mb(static_cast<size_t>(get_formatted_counter_value(gpu_shared_mem_usage_counter_)));
  }

  ///////////////////////////////////////////////////////////////////////
  double get_formatted_counter_value(HCOUNTER counter) const {
    PDH_FMT_COUNTERVALUE counter_value{};
    PDH_STATUS status = PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, nullptr, &counter_value);
    return (status == ERROR_SUCCESS) ? counter_value.doubleValue : 0.0;
  }
};
#endif  // _WIN32
