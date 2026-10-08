// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_compile_model
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, embed_mode_t> > {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_compile_model, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());

  fs::create_directory(temp_dir_);

  fs::path input_model_path = "testdata/mul_1.onnx";
  fs::path output_model_path = temp_dir_ / "mul_1_ctx.onnx";

  network_t network{};
  session_options_t session_options(ov_device_name, network);

  Ort::ModelCompilationOptions compile_options(env_, session_options);
  compile_options.SetEpContextEmbedMode(embed_mode == embed_mode_t::on);
  compile_options.SetInputModelPath(input_model_path.native().c_str());
  compile_options.SetOutputModelPath(output_model_path.native().c_str());

  Ort::CompileModel(env_, compile_options);

  EXPECT_TRUE(fs::exists(output_model_path));

  if (embed_mode == embed_mode_t::off) {
    auto blob_found = [&]() {
      for (const auto& itr : fs::directory_iterator(temp_dir_)) {
        if (fs::is_regular_file(itr.path())) {
          if (itr.path().extension().string() == ".blob" || itr.path().extension().string() == ".bin") {
            return true;
          }
        }
      }

      return false;
    };

    EXPECT_TRUE(blob_found());
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_model_compilation,
    ort_abi_compile_model,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off)),
    [](const testing::TestParamInfo<ort_abi_compile_model::ParamType>& info) { return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)); });
