// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_ep_selection_policy
    : public ortAbiTestFixture,
      public testing::WithParamInterface<OrtExecutionProviderDevicePolicy> {
  std::vector<std::string> additional_required_devices() const override {
    switch (GetParam()) {
      case OrtExecutionProviderDevicePolicy_PREFER_GPU:
        return {"GPU"};
      case OrtExecutionProviderDevicePolicy_PREFER_NPU:
        return {"NPU"};
      default:
        return {};
    }
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_ep_selection_policy, run) {
  const auto& model_info = model_library_t::GetModel("mul_1");

  model_info.create_session(to_string(GetParam()))->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_selection_policy,
    ort_abi_ep_selection_policy,
    testing::Values(
        OrtExecutionProviderDevicePolicy_PREFER_CPU,
        OrtExecutionProviderDevicePolicy_PREFER_GPU,
        OrtExecutionProviderDevicePolicy_PREFER_NPU),
    [](const testing::TestParamInfo<ort_abi_ep_selection_policy::ParamType>& info) { return to_string(info.param); });
