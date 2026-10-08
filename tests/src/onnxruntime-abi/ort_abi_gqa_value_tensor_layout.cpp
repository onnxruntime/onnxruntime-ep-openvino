// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"
#include "ov_constants.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;
using namespace ::onnxruntime::openvino_ep_plugin;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
struct ort_abi_gqa_value_tensor_layout : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////
// gqa_value_tensor_layout API (ORT 1.30)
//
// Check EPs prefered value tensor layout
TEST_P(ort_abi_gqa_value_tensor_layout, check_reported_metadata) {
  device_t device(GetParam());

  const auto metadata_kv_pairs = device.ep_device_.at(0).EpMetadata().GetKeyValuePairs();

  if (const auto& iter = metadata_kv_pairs.find(kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout); iter != metadata_kv_pairs.end()) {
    if (metadata_kv_pairs.contains(std::string(key_ov_meta_device))) {
      OVEP_TESTS_LOG_INFO << "Meta devices are not expected to declare a GQA value tensor layout preference";
      FAIL();
    } else {
      const auto& value = iter->second;
      OVEP_TESTS_LOG_INFO << "\"" << kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout << "\" = \"" << value << "\"";

      const auto hw_device_type = device.ep_device_[0].Device().Type();
      auto preference = ::onnxruntime::openvino_ep_plugin::ep_gqa_value_tensor_layout_preference.at(hw_device_type);
      EXPECT_EQ(value, preference);
    }
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_gqa_value_tensor_layout_physical,
    ort_abi_gqa_value_tensor_layout,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_gqa_value_tensor_layout::ParamType>& info) { return info.param; });

INSTANTIATE_TEST_SUITE_P(
    ort_abi_gqa_value_tensor_layout_virtual,
    ort_abi_gqa_value_tensor_layout,
    testing::Values("AUTO:CPU,GPU"),
    [](const testing::TestParamInfo<ort_abi_gqa_value_tensor_layout::ParamType>& info) {
      std::string result = info.param;
      std::replace(result.begin(), result.end(), ':', '_');
      std::replace(result.begin(), result.end(), ',', '_');
      return result;
    });
