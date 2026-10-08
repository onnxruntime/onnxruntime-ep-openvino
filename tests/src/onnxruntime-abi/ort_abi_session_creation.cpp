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
struct ort_abi_create_session : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_create_session, run) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("mul_1");

  auto session = model_info.create_session(
      ov_device_name,
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_session_creation,
    ort_abi_create_session,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_create_session::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_create_session_from_array : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_create_session_from_array, run) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("mul_1");

  auto session = model_info.create_session(
      ov_device_name,
      network_t::create(model_info.model_path_, session_model_load_mode_t::memory));
  session->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_session_creation,
    ort_abi_create_session_from_array,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_create_session_from_array::ParamType>& info) { return info.param; });
