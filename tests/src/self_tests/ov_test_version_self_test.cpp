// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <sstream>

#include "ov_test_version.h"

using ovep_tests::OvTestVersion;

// Compile-time correctness — all constexpr, verified at build time.
static_assert(OvTestVersion{2026, 0} == OvTestVersion{2026, 0});
static_assert(OvTestVersion{2026, 0} != OvTestVersion{2026, 1});
static_assert(OvTestVersion{2026, 0} < OvTestVersion{2026, 1});
static_assert(OvTestVersion{2026, 1} > OvTestVersion{2026, 0});
static_assert(OvTestVersion{2027, 0} > OvTestVersion{2026, 9});
static_assert(OvTestVersion{2026, 9} < OvTestVersion{2027, 0});
static_assert(OvTestVersion{2026, 0} >= OvTestVersion{2026, 0});
static_assert(OvTestVersion{2026, 1} >= OvTestVersion{2026, 0});
static_assert(OvTestVersion{2026, 0} <= OvTestVersion{2026, 0});
static_assert(OvTestVersion{2026, 0} <= OvTestVersion{2026, 1});
static_assert(!(OvTestVersion{2026, 0} > OvTestVersion{2026, 0}));
static_assert(!(OvTestVersion{2026, 0} < OvTestVersion{2026, 0}));

// built() must satisfy the project's minimum supported version.
static_assert(OvTestVersion::built() >= OvTestVersion{2026, 4});

///////////////////////////////////////////////////////////////////////
TEST(OvTestVersion, stream_output) {
  std::ostringstream os;
  os << OvTestVersion{2026, 1};
  EXPECT_EQ(os.str(), "2026.1");
}

///////////////////////////////////////////////////////////////////////
TEST(OvTestVersion, skip_macro_does_not_skip_current_version) {
  // Passing the built version itself must not trigger a skip.
  const auto v = OvTestVersion::built();
  OVEP_SKIP_IF_OV_BELOW(v.major, v.minor);
  SUCCEED();
}

///////////////////////////////////////////////////////////////////////
TEST(OvTestVersion, skip_macro_skips_impossible_future_version) {
  OVEP_SKIP_IF_OV_BELOW(9999, 0);
  FAIL() << "Should have been skipped";
}
