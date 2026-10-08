// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_external_initializer
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {
  std::vector<std::string> additional_required_devices() const override { return {GetParam()}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_external_initializer, run) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("gemm_external");

  auto session = model_info.create_session(
      ov_device_name,
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_external_initializer,
    ort_abi_external_initializer,
    testing::Values("CPU", "GPU", "NPU", "AUTO:CPU", "AUTO:GPU", "AUTO:NPU", "AUTO:NPU,CPU", "AUTO:NPU,GPU"),
    [](const testing::TestParamInfo<ort_abi_external_initializer::ParamType>& info) {
      std::string result = info.param;
      std::replace(result.begin(), result.end(), ':', '_');
      std::replace(result.begin(), result.end(), ',', '_');
      return result;
    });

///////////////////////////////////////////////////////////////////////
// This test validates that external weights loaded from memory
// using the ORT session option AddExternalInitializersFromFilesInMemory API.
// The external weights binary file is read into a buffer and provided
// to the session options directly in memory, instead of relying on
// file-folder-based external initializer resolution.
///////////////////////////////////////////////////////////////////////
struct ort_abi_external_initializer_from_memory : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_external_initializer_from_memory, run) {
  const auto& ov_device_name = GetParam();
  device_t device(ov_device_name);

  // Copy model and weights to a temp directory so we can delete the weights file
  // without affecting shared test data
  fs::create_directories(temp_dir_);
  fs::copy_file("testdata/gemm_external.onnx", temp_dir_ / "gemm_external.onnx", fs::copy_options::overwrite_existing);
  fs::copy_file("testdata/gemm_external.bin", temp_dir_ / "gemm_external.bin", fs::copy_options::overwrite_existing);

  // Load the model from file on disk (from temp directory)
  network_t network = network_t::create(temp_dir_ / "gemm_external.onnx", session_model_load_mode_t::file);

  // Read the external weights file into a memory buffer
  fs::path weights_path = temp_dir_ / "gemm_external.bin";
  std::vector<char> weights_data = data_utils::read_binary_file<char>(weights_path);

  // Prepare external initializer info for AddExternalInitializersFromFilesInMemory
  std::string weights_filename = weights_path.filename().string();
  std::basic_string<ORTCHAR_T> weights_name_path(weights_filename.begin(), weights_filename.end());
  std::vector<std::basic_string<ORTCHAR_T>> names_path = {weights_name_path};
  std::vector<char*> buffers = {weights_data.data()};
  std::vector<size_t> buffer_sizes = {weights_data.size()};

  // Delete the weights file from disk to ensure weights are loaded purely from memory
  fs::remove(weights_path);
  ASSERT_FALSE(fs::exists(weights_path)) << "Weights file should have been deleted: " << weights_path;

  // Build session options with the device and call AddExternalInitializersFromFilesInMemory
  session_options_t session_options(device, network);
  session_options.AddExternalInitializersFromFilesInMemory(names_path, buffers, buffer_sizes);

  // Create graph (session) using the custom session options
  graph_t graph(network, std::move(session_options));
  pipeline_t pipeline(device, graph);

  // Prepare input/output data
  std::map<std::string, std::vector<std::byte>> input_data;
  input_data.try_emplace("input", data_utils::read_binary_file<std::byte>("testdata/gemm_external.in"));

  std::map<std::string, std::vector<std::byte>> output_data;
  output_data.try_emplace("output", std::vector<std::byte>(graph.output_size("output")));

  // Run inference
  pipeline.run(input_data, output_data);

  // Validate output against reference
  auto reference_output = data_utils::read_binary_file<std::byte>("testdata/gemm_external.out");
  EXPECT_GT(data_utils::cosine_similarity(output_data.at("output"), reference_output), 0.97);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_external_initializer_from_memory,
    ort_abi_external_initializer_from_memory,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_external_initializer_from_memory::ParamType>& info) {
      std::string result = info.param;
      std::replace(result.begin(), result.end(), ':', '_');
      std::replace(result.begin(), result.end(), ',', '_');
      return result;
    });
