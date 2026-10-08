// Copyright (C) Intel Corporation
// Licensed under the MIT License

// Regression tests for the untrusted-compiled-model I/O guards.
//
// The OpenVINO parameter type/shape of the compiled model must not be assumed to
// describe the caller-provided ORT buffer. Fully-static I/O is validated once at
// binding construction (OnnxToOvNetworkBindings). When the ONNX shape is dynamic,
// ORT can't bound the caller's tensor either, so Compute() binds the caller's ORT
// shape and OpenVINO rejects a tensor that doesn't fit the static parameter.
//
// We reproduce the runtime case with reshape_input: it pins the OpenVINO model to
// a fixed [3,2] while the ONNX graph stays dynamic, so ORT accepts a smaller [1,2]
// input and forwards it to the plugin. Binding the [1,2] shape must be rejected
// gracefully (Ort::Exception) instead of reading out of bounds.

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_io_size_validation
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

///////////////////////////////////////////////////////////////////////////
// An input smaller than the OpenVINO [3,2] must be rejected, not over-read.
TEST_P(ort_abi_io_size_validation, undersized_input_is_rejected) {
  device_t device(GetParam());
  if (device.ep_device_.empty()) {
    GTEST_SKIP() << GetParam() << " device not available.";
  }

  // reshape_input pins the OpenVINO model to a static [3,2] while the ONNX
  // graph stays dynamic.
  network_t network{};
  data_utils::read_binary_file("testdata/mul_1_dynamic.onnx", network.buffer_);
  network.ort_provider_options_.emplace("reshape_input", "X[3,2]");
  graph_t graph(device, network);

  // ONNX is dynamic, so ORT accepts this [1,2] input and hands it to the
  // plugin, whose OpenVINO parameter is static [3,2].
  pipeline_t pipeline(device, graph, /*input_shape=*/{1, 2}, /*output_shape=*/{3, 2});

  std::map<std::string, std::vector<float> > input_data;
  input_data.try_emplace("X", std::vector<float>(2, 2.f));
  std::map<std::string, std::vector<float> > output_data;
  output_data.try_emplace("Y", std::vector<float>(6));

  // OpenVINO rejects the [1,2] tensor against its static [3,2] parameter; the exact
  // message is version-dependent, so we only assert a graceful Ort::Exception.
  EXPECT_THROW(pipeline.run(input_data, output_data), Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
// A matching [3,2] input must still run correctly (guard does not over-reject).
TEST_P(ort_abi_io_size_validation, matching_input_runs) {
  device_t device(GetParam());
  if (device.ep_device_.empty()) {
    GTEST_SKIP() << GetParam() << " device not available.";
  }

  network_t network{};
  data_utils::read_binary_file("testdata/mul_1_dynamic.onnx", network.buffer_);
  network.ort_provider_options_.emplace("reshape_input", "X[3,2]");
  graph_t graph(device, network);

  pipeline_t pipeline(device, graph, /*input_shape=*/{3, 2}, /*output_shape=*/{3, 2});

  std::map<std::string, std::vector<float> > input_data;
  input_data.try_emplace("X", std::vector<float>(6, 2.f));
  std::map<std::string, std::vector<float> > output_data;
  output_data.try_emplace("Y", std::vector<float>(6));

  pipeline.run(input_data, output_data);

  EXPECT_THAT(output_data.at("Y"), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_io_size_validation,
    ort_abi_io_size_validation,
    testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
    [](const testing::TestParamInfo<ort_abi_io_size_validation::ParamType>& info) { return info.param; });
