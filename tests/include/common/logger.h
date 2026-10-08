// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "env.h"

namespace ovep_tests {
////////////////////////////////////////////////////////////////////////////////////////
enum log_level_t {
  log_error,
  log_warn,
  log_info,
  log_debug,
  log_func
};

////////////////////////////////////////////////////////////////////////////////////////
inline log_level_t log_level(std::string level) {
  std::map<std::string, log_level_t> map = {
      {"ERROR", log_error},
      {"WARN", log_warn},
      {"INFO", log_info},
      {"DEBUG", log_debug},
      {"FUNCTION", log_func}};

  return map[level];
}

///////////////////////////////////////////////////////////////////////
inline std::ostream& operator<<(std::ostream& os, const log_level_t& level) {
  switch (level) {
    case log_error:
      os << "ERROR";
      break;
    case log_warn:
      os << "WARN";
      break;
    case log_info:
      os << "INFO";
      break;
    case log_debug:
      os << "DEBUG";
      break;
    case log_func:
      os << "FUNC";
      break;
    default:
      break;
  }

  return os;
}
}  // namespace ovep_tests

////////////////////////////////////////////////////////////////////////////////////////
namespace ovep_tests {

////////////////////////////////////////////////////////////////////////////////////////
inline log_level_t current_log_level() {
  static const log_level_t level = []() {
    const auto env = get_environment_variable("OVEP_TESTS_LOG_LEVEL");
    return env.has_value() ? log_level(*env) : log_info;
  }();
  return level;
}

////////////////////////////////////////////////////////////////////////////////////////
template <log_level_t log_level>
struct logger {
  std::ostringstream oss_;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(logger);

  ////////////////////////////////////////////////////////////////////////////////////////
  logger(void) { oss_ << header(log_level); }

  ////////////////////////////////////////////////////////////////////////////////////////
  virtual ~logger(void) {
    if (log_level <= current_log_level()) {
      oss_ << std::endl;
      if constexpr (log_level == log_error) {
        std::cerr << oss_.str();
      } else {
        std::cout << oss_.str();
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  std::ostringstream& stream(void) { return oss_; }

  ////////////////////////////////////////////////////////////////////////////////////////
  std::string header(log_level_t level) {
    std::map<log_level_t, std::string> map = {
        {log_error, "[ ERROR    ] "},
        {log_warn, "[ WARN     ]"},
        {log_info, "[ INFO     ] "},
        {log_debug, "[ DEBUG    ] "},
        {log_func, "[ FUNCTION ] "}};

    return map[level];
  }
};
}  // namespace ovep_tests

////////////////////////////////////////////////////////////////////////////////////////
#define OVEP_TESTS_LOG_ERROR ovep_tests::logger<ovep_tests::log_error>{}.stream()
#define OVEP_TESTS_LOG_WARN ovep_tests::logger<ovep_tests::log_warn>{}.stream()
#define OVEP_TESTS_LOG_INFO ovep_tests::logger<ovep_tests::log_info>{}.stream()
#define OVEP_TESTS_LOG_DEBUG ovep_tests::logger<ovep_tests::log_debug>{}.stream()
#define OVEP_TESTS_LOG_FUNCTION ovep_tests::logger<ovep_tests::log_func>{}.stream()

////////////////////////////////////////////////////////////////////////////////////////
namespace ovep_tests {
////////////////////////////////////////////////////////////////////////////////////////
struct func_guard_t {
  const char* name_ = nullptr;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(func_guard_t);

  ////////////////////////////////////////////////////////////////////////////////////////
  func_guard_t(const char* name) : name_(name) { OVEP_TESTS_LOG_FUNCTION << name_ << ": {"; }

  ////////////////////////////////////////////////////////////////////////////////////////
  virtual ~func_guard_t(void) {
    try {
      OVEP_TESTS_LOG_FUNCTION << name_ << ": ...OK }";
    } catch (...) {
    }
  }
};
}  // namespace ovep_tests
