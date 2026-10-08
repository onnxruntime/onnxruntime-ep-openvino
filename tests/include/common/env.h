// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <optional>
#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace ovep_tests {
inline constexpr char kSkipUnavailableTestsEnvironmentVariable[] = "OVEP_TESTS_SKIP_UNAVAILABLE";

///////////////////////////////////////////////////////////////////////
inline std::optional<std::string> get_environment_variable(const std::string& name) {
#ifdef _WIN32
  DWORD size = GetEnvironmentVariableA(name.c_str(), nullptr, 0);
  if (size == 0) return std::nullopt;
  std::string value(size - 1, '\0');
  GetEnvironmentVariableA(name.c_str(), value.data(), size);
  return value;
#else
  const char* value = getenv(name.c_str());
  if (value == nullptr) return std::nullopt;
  return std::string(value);
#endif
}

///////////////////////////////////////////////////////////////////////
inline bool skip_unavailable_tests_enabled() {
  const auto value = get_environment_variable(kSkipUnavailableTestsEnvironmentVariable);
  return value.has_value() && *value == "1";
}

///////////////////////////////////////////////////////////////////////
inline void set_environment_variable(const std::string& name, const std::string& value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name.c_str(), value.c_str());
#else
  setenv(name.c_str(), value.c_str(), 1);
#endif
}

///////////////////////////////////////////////////////////////////////
inline void unset_environment_variable(const std::string& name) {
#ifdef _WIN32
  SetEnvironmentVariableA(name.c_str(), nullptr);
#else
  unsetenv(name.c_str());
#endif
}

///////////////////////////////////////////////////////////////////////
// Create a temporary environment that lasts for the lifetime of scoped_environment_var.
class scoped_environment_var {
 public:
  scoped_environment_var(const std::string& variable_name, const std::string& value)
      : variable_name_(variable_name), previous_value_(get_environment_variable(variable_name_)) {
    set_environment_variable(variable_name_, value);
  }

  ~scoped_environment_var() {
    if (previous_value_.has_value()) {
      set_environment_variable(variable_name_, *previous_value_);
    } else {
      unset_environment_variable(variable_name_);
    }
  }

  // Delete copy constructor and assignment operator
  scoped_environment_var(const scoped_environment_var&) = delete;
  scoped_environment_var& operator=(const scoped_environment_var&) = delete;

  // Allow move operations
  scoped_environment_var(scoped_environment_var&& other) noexcept
      : variable_name_(std::move(other.variable_name_)), previous_value_(std::move(other.previous_value_)) {
    other.previous_value_ = std::nullopt;
  }

  scoped_environment_var& operator=(scoped_environment_var&& other) noexcept {
    if (this != &other) {
      variable_name_ = std::move(other.variable_name_);
      previous_value_ = std::move(other.previous_value_);
      other.previous_value_ = std::nullopt;
    }
    return *this;
  }

 private:
  std::string variable_name_;
  std::optional<std::string> previous_value_;
};

}  // namespace ovep_tests
