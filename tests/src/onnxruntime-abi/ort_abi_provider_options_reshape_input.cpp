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
struct ort_abi_provider_options_reshape_input
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::pair<std::string, std::string> > {
  std::vector<std::string> additional_required_devices() const override { return {GetParam().first}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_provider_options_reshape_input, run) {
  const auto& ov_device_name = GetParam().first;
  device_t device(ov_device_name);

  network_t network{};
  data_utils::read_binary_file("testdata/mul_1_dynamic.onnx", network.buffer_);
  network.ort_provider_options_.emplace("reshape_input", GetParam().second);

  std::vector<int64_t> shape{3, 2};

  graph_t graph(device, network);
  pipeline_t pipeline(device, graph, shape, shape);

  std::map<std::string, std::vector<float> > input_data;
  input_data.try_emplace("X", std::vector<float>(6, 2.f));

  std::map<std::string, std::vector<float> > output_data;
  output_data.try_emplace("Y", std::vector<float>(6));

  pipeline.run(input_data, output_data);

  EXPECT_THAT(output_data.at("Y"), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_provider_options_reshape_input,
    ort_abi_provider_options_reshape_input,
    testing::Values(
        std::make_pair("CPU", "X[3,2]"),
        std::make_pair("CPU", "X[1..3,1..2]"),
        std::make_pair("GPU", "X[3,2]"),
        std::make_pair("GPU", "X[1..3,1..2]"),
        // std::make_pair("NPU", "X[1..3,1..2]"), // NPU ranged reshape does not currently compile.
        std::make_pair("NPU", "X[3,2]")),
    [](const testing::TestParamInfo<ort_abi_provider_options_reshape_input::ParamType>& info) {
      std::string out = info.param.first + "_shape_" + info.param.second;
      std::replace_if(out.begin(), out.end(), [](char c) { return !std::isalnum(c); }, '_');
      return out;
    });
