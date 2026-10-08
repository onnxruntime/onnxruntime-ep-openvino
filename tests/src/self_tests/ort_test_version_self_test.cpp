// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <sstream>

#include "ort_test_version.h"

using ovep_tests::OrtTestVersion;

// Compile-time correctness for OrtTestVersion comparison operators.
// Uses small arbitrary integers — these are not real ORT versions and
// do not need to be updated when the minimum supported version changes.
static_assert(OrtTestVersion{1} == OrtTestVersion{1});
static_assert(OrtTestVersion{1} != OrtTestVersion{2});
static_assert(OrtTestVersion{1} < OrtTestVersion{2});
static_assert(OrtTestVersion{2} > OrtTestVersion{1});
static_assert(OrtTestVersion{1} >= OrtTestVersion{1});
static_assert(OrtTestVersion{2} >= OrtTestVersion{1});
static_assert(OrtTestVersion{1} <= OrtTestVersion{1});
static_assert(OrtTestVersion{1} <= OrtTestVersion{2});
static_assert(!(OrtTestVersion{1} > OrtTestVersion{1}));
static_assert(!(OrtTestVersion{1} < OrtTestVersion{1}));

// built() must be a positive integer (i.e. parsed from a real ORT header).
static_assert(OrtTestVersion::built().api_version > 0);

///////////////////////////////////////////////////////////////////////
TEST(OrtTestVersion, stream_output) {
  std::ostringstream os;
  os << OrtTestVersion{0};
  EXPECT_EQ(os.str(), "ORT_API_VERSION 0");
}

///////////////////////////////////////////////////////////////////////
TEST(OrtTestVersion, skip_macro_does_not_skip_current_version) {
  const auto v = OrtTestVersion::built();
  OVEP_SKIP_IF_ORT_API_BELOW(v.api_version);
  SUCCEED();
}

///////////////////////////////////////////////////////////////////////
TEST(OrtTestVersion, skip_macro_skips_impossible_future_version) {
  OVEP_SKIP_IF_ORT_API_BELOW(9999);
  FAIL() << "Should have been skipped";
}
