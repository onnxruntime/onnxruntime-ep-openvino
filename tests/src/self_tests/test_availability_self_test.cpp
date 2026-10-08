// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "env.h"

using ovep_tests::scoped_environment_var;

///////////////////////////////////////////////////////////////////////
TEST(TestAvailability, skip_unavailable_requires_exact_one) {
  for (const auto& value : {std::string(""), std::string("0"), std::string("true"), std::string("01")}) {
    scoped_environment_var environment_var(ovep_tests::kSkipUnavailableTestsEnvironmentVariable, value);
    EXPECT_FALSE(ovep_tests::skip_unavailable_tests_enabled()) << "Unexpectedly enabled for value: " << value;
  }

  {
    scoped_environment_var environment_var(ovep_tests::kSkipUnavailableTestsEnvironmentVariable, "1");
    ovep_tests::unset_environment_variable(ovep_tests::kSkipUnavailableTestsEnvironmentVariable);
    EXPECT_FALSE(ovep_tests::skip_unavailable_tests_enabled());
  }

  {
    scoped_environment_var environment_var(ovep_tests::kSkipUnavailableTestsEnvironmentVariable, "1");
    EXPECT_TRUE(ovep_tests::skip_unavailable_tests_enabled());
  }
}
