// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// Tests that inference and EPContext generation work correctly when Unicode
// (non-ASCII) characters appear in:
//   - the EPContext output file path
//   - the model file path
//   - the model filename
//   - the external weights directory

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"
#include "ov_test_version.h"
#include "path_utils.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
// Model with Unicode filename and directory, EPContext output in the same
// Unicode directory. Exercises the full Unicode path: model path, model
// filename, external weights, and EPContext output path.
///////////////////////////////////////////////////////////////////////
struct ort_abi_unicode_epctx
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, embed_mode_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

TEST_P(ort_abi_unicode_epctx, generate_epctx_and_run) {
  const auto& [ov_device_name, embed_mode] = GetParam();
  const auto& model_info = model_library_t::GetModel("gemm_external_unicode");

  const fs::path epctx_dir = temp_dir_ / model_info.model_path_.stem();
  fs::create_directories(epctx_dir);
  const fs::path epctx_model =
      epctx_dir / (epctx_dir.filename() + "_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx");
  const std::string ort_epctx_path = path_to_utf8(epctx_model);

  {
    session_options_wrapper_t session_options(
        ov_device_name,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, ort_epctx_path},
         {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)}});

    model_info.create_session(std::move(session_options));
    ASSERT_TRUE(fs::exists(epctx_model)) << "EPContext not created at: " << ort_epctx_path;
  }

  {
    session_options_wrapper_t session_options(
        ov_device_name,
        {{kOrtSessionOptionEpContextFilePath, ort_epctx_path}});

    auto session = model_info.create_session(std::move(session_options));
    session->run_and_validate();
  }
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_unicode_path,
    ort_abi_unicode_epctx,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off)),
    [](const testing::TestParamInfo<ort_abi_unicode_epctx::ParamType>& info) {
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param));
    });

///////////////////////////////////////////////////////////////////////
// Model whose external initializer bin file has a Unicode filename.
// Exercises weight sharing and OV ONNX frontend external-data resolution
// when the bin file itself is named with non-ASCII characters.
///////////////////////////////////////////////////////////////////////
struct ort_abi_unicode_shared_weights : public ortAbiSingleDeviceTestFixture {};

// Inference with model whose external weights file has a Unicode name.
// Weight sharing enabled so the shared-context manager also processes
// the Unicode bin path.
TEST_P(ort_abi_unicode_shared_weights, infer) {
  const auto& ov_device_name = GetParam();

  const auto& model_info = model_library_t::GetModel("gemm_external_unicode");
  session_options_wrapper_t session_options{ov_device_name, {{kOrtSessionOptionShareEpContexts, "1"}}};
  auto session = model_info.create_session(std::move(session_options));
  session->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_unicode_path,
    ort_abi_unicode_shared_weights,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_unicode_shared_weights::ParamType>& info) {
      return info.param;
    });

///////////////////////////////////////////////////////////////////////
struct ort_abi_unicode_shared_weights_epctx : public ortAbiSingleDeviceTestFixture {};

// EPContext generation with weight sharing using a Unicode-named bin file.
TEST_P(ort_abi_unicode_shared_weights_epctx, generate_epctx_and_run) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("gemm_external_unicode");
  const fs::path epctx_dir = temp_dir_ / model_info.model_path_.stem();
  fs::create_directories(epctx_dir);
  auto epctx_model = epctx_dir / epctx_dir.filename();
  epctx_model += "_ctx_" + ov_device_name + ".onnx";

  {
    session_options_wrapper_t session_options{
        ov_device_name,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model)},
         {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode_t::off)},
         {kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionStopShareEpContexts, "1"}}};

    model_info.create_session(std::move(session_options));
    ASSERT_TRUE(fs::exists(epctx_model)) << "EPContext not created at: " << path_to_utf8(epctx_model);
  }

  {
    session_options_wrapper_t session_options{
        ov_device_name,
        {{kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model)},
         {kOrtSessionOptionShareEpContexts, "1"}}};

    auto session = model_info.create_session(std::move(session_options));
    session->run_and_validate();
  }
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_unicode_path,
    ort_abi_unicode_shared_weights_epctx,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_unicode_shared_weights_epctx::ParamType>& info) {
      return info.param;
    });
