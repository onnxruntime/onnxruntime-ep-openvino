// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
// Test 1: NPU compilation failure
// Validates that exception is thrown when compilation fails on NPU (no
// internal fallback to another OpenVINO device exists).
///////////////////////////////////////////////////////////////////////
struct ort_abi_error_code_npu_compile_failure : public ortAbiSingleDeviceTestFixture {};

TEST_P(ort_abi_error_code_npu_compile_failure, run) {
  const auto& model_info = model_library_t::GetModel("mul_1_dynamic");

  // Use reshape parameter that causes NPU compilation failure
  network_t network = network_t::create(model_info.model_path_);
  network.ort_provider_options_.emplace("reshape_input", "X[1..3,1..2]");

  // Expect exception when NPU compilation fails
  EXPECT_THROW({ auto session = model_info.create_session(GetParam(), std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////
// Test 2: EPContext cross-device import failure
// Validates exception when importing EPContext blob compiled for different device
///////////////////////////////////////////////////////////////////////
struct ort_abi_error_code_epcontext_invalid_import
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::string, embed_mode_t>> {
  std::vector<std::string> additional_required_devices() const override {
    return {std::get<0>(GetParam()), std::get<1>(GetParam())};
  }
};

TEST_P(ort_abi_error_code_epcontext_invalid_import, run) {
  const auto& generate_device = std::get<0>(GetParam());
  const auto& import_device = std::get<1>(GetParam());
  const auto& embed_mode = std::get<2>(GetParam());

  // Skip same-device combinations - they're valid and won't throw
  if (generate_device == import_device) {
    GTEST_SKIP() << "Skipping same-device EPContext import: " << generate_device << " -> " << import_device
                 << " (valid scenario, no failure expected)";
  }

  const auto& model_info = model_library_t::GetModel("mul_1");
  fs::path epctx_model = temp_dir_ / ("mul_1_" + generate_device + "_ctx.onnx");

  // Generate EPContext on first device
  {
    session_options_wrapper_t gen_options(
        generate_device,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
         {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)}});
    model_info.create_session(std::move(gen_options));
    EXPECT_TRUE(fs::exists(epctx_model));
  }

  // Import EPContext on different device - should throw ORT_INVALID_GRAPH exception
  {
    network_t import_network = network_t::create(epctx_model);
    EXPECT_THROW(model_info.create_session(import_device, std::move(import_network)), Ort::Exception);
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_compilation_failure,
    ort_abi_error_code_npu_compile_failure,
    testing::Values("NPU"),
    [](const testing::TestParamInfo<ort_abi_error_code_npu_compile_failure::ParamType>& info) { return info.param + "_compile_failure"; });

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_compilation_failure,
    ort_abi_error_code_epcontext_invalid_import,
    testing::Combine(
        testing::Values("CPU", "GPU", "NPU"),                   // generate_device
        testing::Values("CPU", "GPU", "NPU"),                   // import_device
        testing::Values(embed_mode_t::on, embed_mode_t::off)),  // embed_mode
    [](const testing::TestParamInfo<ort_abi_error_code_epcontext_invalid_import::ParamType>& info) {
      return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) + "_embed_" +
             to_string(std::get<2>(info.param));
    });
