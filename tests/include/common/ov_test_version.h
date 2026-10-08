// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <ostream>

namespace ovep_tests {

struct OvTestVersion {
  int major{};
  int minor{};

  constexpr auto operator<=>(const OvTestVersion&) const noexcept = default;

  static constexpr OvTestVersion built() noexcept {
    return {OVEP_OV_VERSION_MAJOR, OVEP_OV_VERSION_MINOR};
  }
};

inline std::ostream& operator<<(std::ostream& os, OvTestVersion v) {
  return os << v.major << '.' << v.minor;
}

}  // namespace ovep_tests

// Skips the current GTest test if the plugin was built against an OpenVINO
// version older than (maj, min).
#define OVEP_SKIP_IF_OV_BELOW(maj, min)                                               \
  do {                                                                                \
    if (::ovep_tests::OvTestVersion::built() < ::ovep_tests::OvTestVersion{maj, min}) \
      GTEST_SKIP() << "Requires OV " << (maj) << "." << (min)                         \
                   << "+, built against " << ::ovep_tests::OvTestVersion::built();    \
  } while (false)
