// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_enumerate_ov_device : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_enumerate_ov_device, run) {
  const auto& ov_device_name = GetParam();
  device_t device(ov_device_name);

  const auto& ep_devices = device.ep_device_;
  EXPECT_GT(ep_devices.size(), 0);

  if (ep_devices.size() > 0) {
    const auto ep_metadata = ep_devices[0].EpMetadata();
    EXPECT_GT(ep_metadata.GetKeyValuePairs().size(), 0);

    if (ep_metadata.GetKeyValuePairs().size() > 0) {
      EXPECT_EQ(ep_metadata.GetValue("ov_device"), ov_device_name);

      // Only do the driver version check on Plugin EP
      if (loaded_ep_ == ep_type_t::abi && ov_device_name == "NPU") {
        EXPECT_GT(std::stoi(ep_metadata.GetValue(kOrtEpDevice_EpMetadataKey_OSDriverVersion)), 1000000);
      }

      // Weightless is NPU only, and this is the discovery-time counterpart of GetWeightlessSupport
      if (loaded_ep_ == ep_type_t::abi) {
        EXPECT_STREQ(ep_metadata.GetValue(kOrtEpDevice_EpMetadataKey_WeightlessSupport),
                     ov_device_name == "NPU" ? "external_only" : "none");
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_device_enumeration,
    ort_abi_enumerate_ov_device,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_enumerate_ov_device::ParamType>& info) { return info.param; });
