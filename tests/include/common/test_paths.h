// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace ovep_tests {

inline fs::path GetProcessTempDir() {
#ifdef _WIN32
  const auto process_id = _getpid();
#else
  const auto process_id = getpid();
#endif
  return fs::path("temp") / ("process_" + std::to_string(process_id));
}

}  // namespace ovep_tests
