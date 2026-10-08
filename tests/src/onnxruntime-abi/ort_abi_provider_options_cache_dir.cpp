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
struct ort_abi_provider_options_cache_dir : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_provider_options_cache_dir, run) {
  const auto& ov_device_name = GetParam();
  device_t device(ov_device_name);

  const std::string cache_dir = path_to_utf8(temp_dir_);

  network_t network{};
  data_utils::read_binary_file("testdata/mul_1.onnx", network.buffer_);
  network.ort_provider_options_load_config_.emplace("CACHE_DIR", cache_dir);

  graph_t graph(device, network);

  auto blob_file_found = [&](void) {
    if (!std::filesystem::exists(cache_dir) || !std::filesystem::is_directory(cache_dir)) {
      return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
      if (std::filesystem::is_regular_file(entry.path())) {
        if (entry.path().extension() == ".blob") {
          return true;
        }
      }
    }

    return false;
  };

  EXPECT_TRUE(blob_file_found());
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_provider_options_cache_dir,
    ort_abi_provider_options_cache_dir,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_provider_options_cache_dir::ParamType>& info) { return info.param; });
