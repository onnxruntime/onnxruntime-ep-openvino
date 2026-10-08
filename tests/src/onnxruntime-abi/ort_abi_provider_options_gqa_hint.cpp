// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

using namespace ovep_tests;

using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;

struct ort_abi_provider_options_gqa_hint : public ortAbiSingleDeviceTestFixture {};

TEST_P(ort_abi_provider_options_gqa_hint, accepts_valid_values_for_non_gqa_model) {
  device_t device(GetParam());
  for (const auto* value : {"", "true", "false", "TRUE", "FALSE", "1", "0"}) {
    SCOPED_TRACE(value);
    auto network = network_t::create("testdata/mul_1.onnx");
    if (*value != '\0') {
      network.ort_provider_options_["hint_enable_gqa"] = value;
    }
    EXPECT_NO_THROW({ graph_t graph(device, network); });
  }
}

TEST_P(ort_abi_provider_options_gqa_hint, rejects_invalid_values) {
  device_t device(GetParam());
  for (const auto* value : {"yes", "no", "invalid"}) {
    SCOPED_TRACE(value);
    auto network = network_t::create("testdata/mul_1.onnx");
    network.ort_provider_options_["hint_enable_gqa"] = value;
    try {
      graph_t graph(device, network);
      ADD_FAILURE() << "Expected an invalid provider option error";
    } catch (const Ort::Exception& e) {
      // ORT may wrap the EP's ORT_INVALID_ARGUMENT status as ORT_FAIL; the message is preserved.
      EXPECT_THAT(e.what(), testing::HasSubstr("hint_enable_gqa"));
      EXPECT_THAT(e.what(), testing::HasSubstr("Valid values are: 'true', 'false', '1', '0'"));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ort_abi_provider_options_gqa_hint, ort_abi_provider_options_gqa_hint,
                         ortAbiSingleDeviceTestFixture::all_devices());