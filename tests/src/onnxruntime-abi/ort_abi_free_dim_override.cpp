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
struct ort_abi_free_dim_override : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_free_dim_override, run) {
  const auto& ov_device_name = GetParam();
  device_t device(ov_device_name);

  network_t network{};
  data_utils::read_binary_file("testdata/mul_1_dynamic.onnx", network.buffer_);
  network.ort_free_dim_override_.insert({{"i_0", 3}, {"i_1", 2}});

  graph_t graph(device, network);
  pipeline_t pipeline(device, graph);

  std::map<std::string, std::vector<float> > input_data;
  input_data.try_emplace("X", std::vector<float>(6, 2.f));

  std::map<std::string, std::vector<float> > output_data;
  output_data.try_emplace("Y", std::vector<float>(6));

  pipeline.run(input_data, output_data);

  EXPECT_THAT(output_data.at("Y"), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_free_dim_override,
    ort_abi_free_dim_override,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_free_dim_override::ParamType>& info) { return info.param; });
