// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <charconv>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <array>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <format>
#include <thread>
#include <type_traits>
#include <string>
#include <string_view>
#include <algorithm>

#include "onnxruntime/onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ort_api_version.h"

#define OVEP_ERROR_STR(...) onnxruntime::openvino_ep_plugin::BuildErrorMessage(__FILE__, __LINE__, __VA_ARGS__)

void handle_invalid_status();

template <class>
inline constexpr bool always_false_v = false;

#define OVEP_RETURN_IF_ERROR(fn) \
  do {                           \
    OrtStatus* _status = (fn);   \
    if (_status != nullptr) {    \
      return _status;            \
    }                            \
  } while (0)

#define OVEP_RETURN_IF(cond, ort_api, msg)                                     \
  do {                                                                         \
    if ((cond)) {                                                              \
      return (ort_api).CreateStatus(ORT_EP_FAIL, OVEP_ERROR_STR(msg).c_str()); \
    }                                                                          \
  } while (0)

#define OVEP_RETURN_STATUS_IF(cond, status, msg)                              \
  do {                                                                        \
    if ((cond)) {                                                             \
      return Ort::GetApi().CreateStatus(status, OVEP_ERROR_STR(msg).c_str()); \
    }                                                                         \
  } while (0)

#define OVEP_DISABLE_MOVE(class_name) \
  class_name(class_name&&) = delete;  \
  class_name& operator=(class_name&&) = delete;

#define OVEP_DISABLE_COPY(class_name)     \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete;

#define OVEP_DISABLE_COPY_AND_MOVE(class_name) \
  OVEP_DISABLE_COPY(class_name)                \
  OVEP_DISABLE_MOVE(class_name)

#define OVEP_THROW(...)                                    \
  do {                                                     \
    throw std::runtime_error(OVEP_ERROR_STR(__VA_ARGS__)); \
  } while (0)

#define OVEP_ENFORCE(condition, ...) \
  do {                               \
    if (!(condition)) {              \
      OVEP_THROW(__VA_ARGS__);       \
    }                                \
  } while (0)

#define OVEP_MAKE_STATUS(...) Ort::Status(__VA_ARGS__).release()

#define OVEP_API_ENTRY(derived_type, member_func)                                           \
  onnxruntime::openvino_ep_plugin::ApiEntry<derived_type, &derived_type::member_func,       \
                                            std::to_array(#derived_type "::" #member_func), \
                                            ORT_RUNTIME_EXCEPTION, std::to_array(ORT_FILE), __LINE__>

#define OVEP_API_ENTRY_EX(derived_type, member_func, error_code)                            \
  onnxruntime::openvino_ep_plugin::ApiEntry<derived_type, &derived_type::member_func,       \
                                            std::to_array(#derived_type "::" #member_func), \
                                            error_code, std::to_array(ORT_FILE), __LINE__>

namespace onnxruntime {
namespace openvino_ep_plugin {

struct OvLoggable {
  OvLoggable(const OrtLogger& logger) : logger_{Ort::Logger{&logger}} {}
  OvLoggable(Ort::Logger logger) : logger_{logger} {}
  Ort::Logger GetLogger() const { return logger_; }
  Ort::Logger logger_;
};

constexpr const char* kOvepLogTag = "[OpenVINO-EP] ";
constexpr const char* kOpenVINOExecutionProvider = "OpenVINOExecutionProvider";

template <typename... Args>
std::string BuildErrorMessage(const char* file, int line, Args&&... args) {
  std::ostringstream oss;
  oss << kOvepLogTag;
  oss << file << ":" << line << " - ";
  ((oss << args), ...);
  return std::move(oss).str();
}

template <typename T>
concept HasLogger = requires(T t) {
  { t.GetLogger() } -> std::same_as<Ort::Logger>;
};

template <typename T>
concept IsOstreamWritable = requires(std::ostream& oss, const T& value) {
  { oss << value };
};

template <typename T>
std::string ArgToString(const T& value) {
  using arg_type = std::decay_t<T>;
  if constexpr (std::is_same_v<arg_type, const char*> || std::is_same_v<arg_type, char*>) {
    return value ? std::string(value) : "<null>";
  } else if constexpr (std::is_pointer_v<arg_type>) {
    std::ostringstream oss;
    oss << static_cast<const void*>(value);
    return std::move(oss).str();
  } else if constexpr (IsOstreamWritable<arg_type>) {
    std::ostringstream oss;
    oss << value;
    return std::move(oss).str();
  } else {
    return "<non-streamable>";
  }
}

template <typename... Args>
std::string BuildArgsLogString(const Args&... args) {
  std::ostringstream oss;
  oss << "(";
  bool first = true;
  [[maybe_unused]] auto append_arg = [&](const auto& arg) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << ArgToString(arg);
  };
  (append_arg(args), ...);
  oss << ")";
  return std::move(oss).str();
}

// Helper for handling status returns in ApiEntry
template <typename ReturnType>
auto MakeErrorReturn(const char* msg, OrtErrorCode code) noexcept {
  if constexpr (std::is_void_v<ReturnType>) {
    return;
  } else if constexpr (std::is_same_v<ReturnType, OrtStatus*>) {
    return Ort::Status(msg, code).release();
  } else {
    return ReturnType{};
  }
}

/// @brief RAII trace logger that logs ENTRY on construction and EXIT (with elapsed time) on destruction.
/// Works with the ORT Logger API. Used by both the OVEP_TRACE macro and ApiEntry.
class OvepTraceLog {
 public:
  // No-detail constructor: logs "ENTRY [tid:<id>]"
  OvepTraceLog(const Ort::Logger& logger,
               const ORTCHAR_T* file,
               int line,
               const char* func) noexcept
      : logger_(logger),
        active_(ORT_LOGGING_LEVEL_VERBOSE >= logger.GetLoggingSeverityLevel()),
        file_(file),
        line_(line),
        func_(func),
        start_(std::chrono::steady_clock::now()) {
    if (!active_) return;
    tid_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
    LogFormatted("ENTRY [tid:%zu]", tid_);
  }

  // Detail constructor (no format args): logs "ENTRY [tid:<id>] [<detail>]"
  OvepTraceLog(const Ort::Logger& logger,
               const ORTCHAR_T* file,
               int line,
               const char* func,
               const char* detail) noexcept
      : logger_(logger),
        active_(ORT_LOGGING_LEVEL_VERBOSE >= logger.GetLoggingSeverityLevel()),
        file_(file),
        line_(line),
        func_(func),
        start_(std::chrono::steady_clock::now()) {
    if (!active_) return;
    tid_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
    LogFormatted("ENTRY [tid:%zu] [%s]", tid_, detail);
  }

  // Detail constructor with printf-style format string: logs "ENTRY [tid:<id>] [<detail>]"
  template <typename... Args>
    requires(sizeof...(Args) > 0)
  OvepTraceLog(const Ort::Logger& logger,
               const ORTCHAR_T* file,
               int line,
               const char* func,
               const char* format,
               Args&&... args) noexcept
      : logger_(logger),
        active_(ORT_LOGGING_LEVEL_VERBOSE >= logger.GetLoggingSeverityLevel()),
        file_(file),
        line_(line),
        func_(func),
        start_(std::chrono::steady_clock::now()) {
    if (!active_) return;
    tid_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
    char detail_buf[448];
    std::snprintf(detail_buf, sizeof(detail_buf), format, args...);
    LogFormatted("ENTRY [tid:%zu] [%s]", tid_, detail_buf);
  }

  ~OvepTraceLog() noexcept {
    if (!active_) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start_)
                  .count();
    LogFormatted("EXIT [tid:%zu] (%lld ms)", tid_, static_cast<long long>(ms));
  }

  OvepTraceLog(const OvepTraceLog&) = delete;
  OvepTraceLog& operator=(const OvepTraceLog&) = delete;

 private:
  template <typename... Args>
  void LogFormatted(const char* format, Args&&... args) const noexcept {
    static_cast<void>(logger_.LogFormattedMessage(ORT_LOGGING_LEVEL_VERBOSE, file_, line_, func_, format, args...));
  }

  const Ort::Logger& logger_;
  bool active_;
  const ORTCHAR_T* file_;
  int line_;
  const char* func_;
  size_t tid_{0};
  std::chrono::steady_clock::time_point start_;
};

#define OVEP_TRACE_CONCAT_INNER(a, b) a##b
#define OVEP_TRACE_CONCAT(a, b) OVEP_TRACE_CONCAT_INNER(a, b)

#define OVEP_TRACE(logger, ...)                                                                  \
  onnxruntime::openvino_ep_plugin::OvepTraceLog OVEP_TRACE_CONCAT(_ovep_trace_guard_, __LINE__)( \
      (logger), ORT_FILE, __LINE__, static_cast<const char*>(__FUNCTION__), ##__VA_ARGS__)

// Template for creating C API callback thunks from C++ member functions.
// Automatically handles casting, exception handling, logger propagation, and error codes.
// Uses OvepTraceLog for RAII-based entry/exit tracing when the Derived type has a logger.
// Set DisableLog=true for teardown paths where ORT logging infrastructure may already be invalid.
template <typename Derived, auto MemberFunc, auto FuncName, OrtErrorCode ErrorCode = ORT_RUNTIME_EXCEPTION,
          auto FileName, int FileLine = 0, bool DisableLog = false>
auto ORT_API_CALL ApiEntry(auto* this_ptr, auto... args) noexcept {
  using this_type = std::remove_pointer_t<decltype(this_ptr)>;
  using qualified_derived = std::conditional_t<std::is_const_v<this_type>, const Derived, Derived>;
  using return_type = decltype((std::declval<qualified_derived*>()->*MemberFunc)(args...));

  auto* derived = static_cast<qualified_derived*>(this_ptr);
  std::optional<Ort::Logger> logger;

  auto try_log_error_and_return_val = [&](const char* msg, OrtErrorCode code = ErrorCode) {
    if (logger && ORT_LOGGING_LEVEL_ERROR >= logger->GetLoggingSeverityLevel()) {
      static_cast<void>(logger->LogMessage(ORT_LOGGING_LEVEL_ERROR, FileName.data(), FileLine,
                                           FuncName.data(), msg));
    }
    return MakeErrorReturn<return_type>(msg, code);
  };

  try {
    std::optional<OvepTraceLog> trace_guard;
    if constexpr (HasLogger<Derived> && !DisableLog) {
      logger = derived->GetLogger();

      if (ORT_LOGGING_LEVEL_VERBOSE >= logger->GetLoggingSeverityLevel()) {
        const auto args_str = BuildArgsLogString(args...);
        trace_guard.emplace(*logger, FileName.data(), FileLine,
                            FuncName.data(),
                            "%s", args_str.c_str());
      }
    }

    return (derived->*MemberFunc)(args...);

  } catch (const Ort::Exception& ex) {
    return try_log_error_and_return_val(ex.what(), ex.GetOrtErrorCode());
  } catch (const std::exception& ex) {
    return try_log_error_and_return_val(ex.what());
  } catch (...) {
    return try_log_error_and_return_val("Unknown exception occurred.");
  }
}

#define OVEP_API_ENTRY_TEARDOWN(derived_type, member_func)                                  \
  onnxruntime::openvino_ep_plugin::ApiEntry<derived_type, &derived_type::member_func,       \
                                            std::to_array(#derived_type "::" #member_func), \
                                            ORT_RUNTIME_EXCEPTION, std::to_array(ORT_FILE), __LINE__, true>

inline bool IsPowerOfTwo(size_t value) {
  return value && ((value & (value - 1)) == 0);
}

inline size_t AlignUp(size_t value, size_t alignment) {
  OVEP_ENFORCE(IsPowerOfTwo(alignment), "Alignment must be a power of two.");
  return (value + alignment - 1) & ~(alignment - 1);
}

// Checks offset + size <= total_size without overflow
inline bool IsWithinBounds(size_t offset, size_t size, size_t total_size) {
  return (offset < total_size) && (size <= total_size) && (offset <= total_size - size);
}

// Construct a std::filesystem::path from a UTF-8 encoded string.
// On Windows, std::filesystem::path(std::string) uses the ANSI code page;
// this helper avoids that by using the explicit UTF-8 construction.
inline std::filesystem::path PathFromUtf8(std::string_view utf8) {
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

// Convert a std::filesystem::path to a UTF-8 encoded std::string.
// On Windows, path::string() returns a string in the ANSI code page;
// this helper always returns UTF-8.
inline std::string ToUtf8String(const std::filesystem::path& path) {
  auto u8str = path.u8string();
  return {u8str.begin(), u8str.end()};
}

// True if `relative_path` is a non-empty relative path with no root or ".." traversal, i.e. safe to
// join beneath a trusted base directory.
inline bool IsSafeRelativeSubpath(const std::filesystem::path& relative_path) {
  if (relative_path.empty() || relative_path.is_absolute() || relative_path.has_root_name() ||
      relative_path.has_root_directory()) {
    return false;
  }
  const auto normalized = relative_path.lexically_normal();
  return std::none_of(normalized.begin(), normalized.end(),
                      [](const std::filesystem::path& part) { return part == ".."; });
}

std::string GetEnvironmentVar(std::string_view env_var);

// Parses the ORT API version from a version string of the form "1.{API_VERSION}.*".
// Returns nullopt if the string is null, malformed, or does not match the expected format.
inline std::optional<uint32_t> TryGetAPIVersionFromVersionString(const char* version_str) {
  if (!version_str) {
    return std::nullopt;
  }
  const std::string_view sv(version_str);
  if (!sv.starts_with("1.")) {
    return std::nullopt;
  }
  const std::string_view rest = sv.substr(2);
  const auto dot = rest.find('.');
  if (dot == std::string_view::npos) {
    return std::nullopt;
  }
  uint32_t version = 0;
  const auto [ptr, ec] = std::from_chars(rest.data(), rest.data() + dot, version);
  if (ec != std::errc{} || ptr != rest.data() + dot) {
    return std::nullopt;
  }
  return version;
}

inline void InitCxxApi(const OrtApiBase& ort_api_base) {
  static std::once_flag init_api;
  std::call_once(init_api, [&]() {
    const char* const version_str = ort_api_base.GetVersionString();
    const auto parsed_version = TryGetAPIVersionFromVersionString(version_str);
    OVEP_ENFORCE(parsed_version.has_value(),
                 std::format("Failed to parse ONNX Runtime version string '{}'.",
                             version_str ? version_str : "null"));

    ort_runtime_api_version = std::min<uint32_t>(*parsed_version, ort_compile_time_api_version);
    OVEP_ENFORCE(ort_runtime_api_version >= ort_min_api_version_supported,
                 std::format("ONNX Runtime version '{}' (parsed API version: {}) is below the minimum "
                             "supported version {}.",
                             version_str, ort_runtime_api_version, ort_min_api_version_supported));

    const OrtApi* const ort_api = ort_api_base.GetApi(ort_runtime_api_version);
    OVEP_ENFORCE(ort_api != nullptr,
                 std::format("ONNX Runtime version '{}' does not support the parsed API version {}.",
                             version_str, ort_runtime_api_version));
    Ort::InitApi(ort_api);
  });
}

inline OrtStatus* InitCxxApiOrStatus(const OrtApiBase* ort_api_base) {
  // GetApi(1) is expected to always succeed on any valid OrtApiBase.
  try {
    InitCxxApi(*ort_api_base);
    return nullptr;
  } catch (const std::exception& e) {
    return ort_api_base->GetApi(1)->CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unknown exception during ORT API initialization.");
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
