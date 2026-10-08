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

///////////////////////////////////////////////////////////////////////
struct ort_abi_provider_options_dump_subgraphs : public ortAbiTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_provider_options_dump_subgraphs, run) {
  device_t device("CPU");

  network_t network{};
  data_utils::read_binary_file("testdata/mul_1.onnx", network.buffer_);

  const std::filesystem::path dump_dir = temp_dir_ / "dump_subgraphs";
  network.ort_provider_options_["dump_subgraphs"] = dump_dir.string();

  graph_t graph(device, network);

  ASSERT_TRUE(std::filesystem::exists(dump_dir))
      << "Expected dump_subgraphs to create directory: " << dump_dir.string();

  bool transformed_input_dump_found = false;

  for (const auto& entry : std::filesystem::directory_iterator(dump_dir)) {
    if (!std::filesystem::is_regular_file(entry.path())) {
      continue;
    }

    const std::string file_name = entry.path().filename().string();
    if (file_name.ends_with("_after_transformation_input_model.xml")) {
      transformed_input_dump_found = true;
    }
  }

  EXPECT_TRUE(transformed_input_dump_found)
      << "Expected transformed input model dump file in: " << dump_dir.string();
}