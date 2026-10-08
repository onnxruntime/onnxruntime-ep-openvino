// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

namespace ovep_tests {
namespace Runner {
template <typename TestT>
void Execute(TestT&& test) {
  test.Setup();
  test.Run();

  EXPECT_TRUE(test.CheckResults());

  test.Teardown();
}
}  // namespace Runner
}  // namespace ovep_tests