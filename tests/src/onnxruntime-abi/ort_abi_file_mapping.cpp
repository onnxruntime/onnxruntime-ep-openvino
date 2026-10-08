// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#ifdef _WIN32

#include "ortAbiTestFixture.h"
#include "memory.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

struct ep_ctx_scenarios : public ortAbiSingleDeviceTestFixture {
  void SetUp() override {
    EmptyWorkingSet(GetCurrentProcess());
    ortAbiTestFixture::SetUp();
  }

  void RunTest(bool share_ep_contexts) {
    const auto& ov_device_name = GetParam();
    device_t device(ov_device_name);

    fs::create_directories(temp_dir_);
    fs::path test_data = "testdata";
    fs::path orig_model_path = test_data / "gemm_external.onnx";
    fs::path ep_context = temp_dir_ / "ep_context.onnx";

    auto find_bin_file = [](const fs::path& model_path) {
      // iterate through files in directory to find file that ends with .bin
      for (const auto& entry : fs::directory_iterator(model_path.parent_path())) {
        if (entry.path().extension() == ".bin" &&
            path_to_utf8(entry.path().stem()).contains(path_to_utf8(model_path.stem()))) {
          return entry.path();
        }
      }
      return fs::path{};
    };

    // generate ep context
    {
      auto network = network_t::create(orig_model_path);
      network.ort_session_config_[kOrtSessionOptionEpContextEnable] = "1";
      network.ort_session_config_[kOrtSessionOptionEpContextEmbedMode] = "0";
      network.ort_session_config_[kOrtSessionOptionEpContextFilePath] = ep_context.string();
      if (share_ep_contexts) {
        network.ort_session_config_[kOrtSessionOptionShareEpContexts] = "1";
      }

      graph_t graph(device, network);
      fs::copy_file(
          test_data / "gemm_external.bin",
          temp_dir_ / "gemm_external.bin",
          fs::copy_options::overwrite_existing);
    }

    std::map<std::string, std::vector<std::byte> > input_data;
    std::vector<std::byte> input_bytes =
        data_utils::read_binary_file<std::byte>(test_data / "gemm_external.in");
    input_data.try_emplace("input", input_bytes);
    std::map<std::string, std::vector<std::byte> > output_data;
    output_data.try_emplace(
        "output",
        std::vector<std::byte>(fs::file_size(test_data / "gemm_external.out")));

    cpu_mem_counter_t mem_counter;
    mem_counter.start();

    auto network = network_t::create(ep_context);  // Load to ensure file is created
    if (share_ep_contexts) {
      network.ort_session_config_[kOrtSessionOptionShareEpContexts] = "1";
    }
    graph_t graph(device, network);
    pipeline_t pipeline(device, graph);
    pipeline.run(input_data, output_data);

    auto load_mem_usage = mem_counter.delta_from_start();
    OVEP_TESTS_LOG_INFO << "Delta " << load_mem_usage;

    const bool verify_virtual_memory = ov_device_name == "NPU";
    if (share_ep_contexts) {
      auto weights_size = memory_utils::byte_to_mb(fs::file_size(find_bin_file(orig_model_path)));

      // for GPU the weight file is unmapped, so the resident mem share will be lower...
      if (ov_device_name != "GPU") {
        EXPECT_GE(load_mem_usage.resident_shared_memory_, weights_size);
      }
      if (verify_virtual_memory) {
        EXPECT_LT(load_mem_usage.virtual_memory_, weights_size);
      }
    } else {
      auto bin_size = memory_utils::byte_to_mb(fs::file_size(find_bin_file(ep_context)));
      EXPECT_GE(load_mem_usage.resident_shared_memory_, bin_size);
      if (verify_virtual_memory) {
        EXPECT_LT(load_mem_usage.virtual_memory_, bin_size);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ep_ctx_scenarios, native_blob_mapped) {
  RunTest(false);
}

///////////////////////////////////////////////////////////////////////////
TEST_P(ep_ctx_scenarios, shared_weights_mapped) {
  RunTest(true);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_file_mapping,
    ep_ctx_scenarios,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const auto& info) { return info.param; });

#endif  // _WIN32
