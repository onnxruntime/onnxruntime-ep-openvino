// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <onnxruntime/onnxruntime_c_api.h>

#include <ostream>

namespace ovep_tests {

struct OrtTestVersion {
  int api_version{};

  constexpr auto operator<=>(const OrtTestVersion&) const noexcept = default;

  static constexpr OrtTestVersion built() noexcept {
    return {ORT_API_VERSION};
  }
};

inline std::ostream& operator<<(std::ostream& os, OrtTestVersion v) {
  return os << "ORT_API_VERSION " << v.api_version;
}

}  // namespace ovep_tests

// Skips the current GTest test if the plugin was built against an ONNX Runtime
// version with ORT_API_VERSION older than (ver).
#define OVEP_SKIP_IF_ORT_API_BELOW(ver)                                            \
  do {                                                                             \
    if (::ovep_tests::OrtTestVersion::built() < ::ovep_tests::OrtTestVersion{ver}) \
      GTEST_SKIP() << "Requires ORT_API_VERSION >= " << (ver)                      \
                   << ", built against " << ::ovep_tests::OrtTestVersion::built(); \
  } while (false)
