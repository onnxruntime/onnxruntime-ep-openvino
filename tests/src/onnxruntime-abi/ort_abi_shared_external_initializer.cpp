// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// This file is Windows-only due to platform-specific GPU memory counter functionality
#ifdef _WIN32

#include "ortAbiTestFixture.h"
#include "memory.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;

struct ort_abi_shared_external_initializer_fixture : public ortAbiSingleDeviceTestFixture {
  std::optional<gpu_mem_counter_t> try_get_device_counter(device_t& device) {
    std::optional<gpu_mem_counter_t> counter;
    if (device.name_ == "NPU" || device.name_ == "GPU") {
      counter.emplace(device_t::get_device_luid(device.ep_device_.at(0)));
    } else {
      OVEP_TESTS_LOG_INFO << "Skipping device memory usage checks for " << device.name_;
    }
    return counter;
  }

  const auto& get_model_info() { return model_library_t::GetModel("gemm_external"); }

  fs::path get_external_data_file() { return get_model_info().external_data_path_; }
  auto get_external_weights_size() { return fs::file_size(get_external_data_file()); }
};

struct device_parameterized : public ort_abi_shared_external_initializer_fixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(device_parameterized, run_without_ep_context) {
  constexpr size_t iterations = 1;

  std::vector<std::unique_ptr<builtin_session_t>> sessions(iterations);
  const auto& ov_device_name = GetParam();

  // Checks removed because memory savings aren't realized in this path.
  // This test effectively just makes sure that we are functional if weight sharing is set.
  // auto gpu_mem_counter = try_get_device_counter( device );
  // if( gpu_mem_counter )
  // {
  //     gpu_mem_counter->start();
  // }

  for (auto& session : sessions) {
    session_options_t session_options{
        ov_device_name, {{kOrtSessionOptionShareEpContexts, "1"}}};

    session = get_model_info().create_session(std::move(session_options));
    session->run_and_validate();
  }

  // if( gpu_mem_counter )
  // {
  //     gpu_mem_counter->stop();
  //     EXPECT_LT(
  //         gpu_mem_counter->gpu_shared_mem_delta_,
  //         BYTE_TO_MB( fs::file_size( "testdata/gemm_external.bin" ) ) * ( iterations / 2 ) );
  // }
}

///////////////////////////////////////////////////////////////////////////
TEST_P(device_parameterized, ep_ctx_multi_load) {
  constexpr size_t iterations = 50;

  const auto& ov_device_name = GetParam();
  auto device = device_t(ov_device_name);

  const auto epctx_model = path_to_utf8(temp_dir_ / ("gemm_external_ctx_" + ov_device_name + ".onnx"));

  // Generate epctx model
  {
    session_options_t session_options{
        ov_device_name,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, epctx_model},
         {kOrtSessionOptionShareEpContexts, "1"}}};

    get_model_info().create_session(std::move(session_options));

    ASSERT_TRUE(fs::exists(epctx_model));
  }

  // Run epctx model
  {
    auto gpu_mem_counter = try_get_device_counter(device);

    std::vector<std::unique_ptr<builtin_session_t>> sessions;
    sessions.reserve(iterations);

    bool first_iteration = true;
    for (size_t idx = 0; idx < iterations; ++idx) {
      session_options_t session_options{
          ov_device_name,
          {{kOrtSessionOptionShareEpContexts, "1"}}};

      auto session = get_model_info().create_session(std::move(session_options));
      session->run_and_validate();
      if (first_iteration && gpu_mem_counter) {
        // skip the allocations of the first session as it can include extra data
        gpu_mem_counter->start();
        first_iteration = false;
      }
      sessions.push_back(std::move(session));
    }

    if (gpu_mem_counter) {
      auto delta = gpu_mem_counter->delta_from_start();
      EXPECT_LT(
          delta.shared_mem_usage_ / (iterations - 1),  // per session marginal memory increase should be < the weights
          memory_utils::byte_to_mb(get_external_weights_size()));
    }
  }
}

// This test validates the weights-as-input conversion pass when weight sharing is enabled.
// Scales and zero-point QDQ constants must not be converted to inputs; otherwise, compilation will fail.
// Accuracy is checked only on CPU; for other devices, the test ensures successful compilation and execution. This is
// because at the time of test writing it was difficult to assert accuracy across devices. NPU is skipped below.
TEST_P(device_parameterized, ep_ctx_load_qdq_external_scale_and_zp) {
  if (GetParam() == "NPU") {
    // vpux-compiler bug: NCE output-channel alignment pads the per-axis quant scale array but not
    // the weights-as-input buffer shape, since this model's 1000 output channels aren't 16-aligned.
    GTEST_SKIP() << "Skipping NPU: known vpux-compiler bug with non-16-aligned per-channel "
                    "quantized weights-as-input (scale count vs weight buffer shape mismatch)";
  }

  const auto& model_info = model_library_t::GetModel("gemm_qdq_scale_zp_external");

  session_options_t session_options{
      GetParam(),
      {{kOrtSessionOptionShareEpContexts, "1"}}};

  auto session = model_info.create_session(std::move(session_options));
  session->run_and_validate();
}

TEST_P(device_parameterized, multi_set_shared_external_weights) {
  constexpr size_t set_size = 50;

  const auto& ov_device_name = GetParam();
  device_t device(ov_device_name);

  auto get_ep_context_path = [](size_t set_index) -> fs::path {
    return temp_dir_ / std::format("ep_ctx_{}.onnx", set_index + 1);
  };

  {
    // Generate shared ep context sets
    for (size_t set_index = 0; set_index < set_size; ++set_index) {
      auto ep_context_path = get_ep_context_path(set_index);

      session_options_t session_options{
          device_t(ov_device_name),
          {{kOrtSessionOptionShareEpContexts, "1"},
           {kOrtSessionOptionEpContextEnable, "1"},
           {kOrtSessionOptionStopShareEpContexts, "1"},
           {kOrtSessionOptionEpContextFilePath, path_to_utf8(ep_context_path)}}};

      get_model_info().create_session(std::move(session_options));
      ASSERT_TRUE(fs::exists(ep_context_path));
    }
  }

  std::vector<std::unique_ptr<builtin_session_t>> sessions;
  sessions.reserve(set_size);

  auto counter = try_get_device_counter(device);
  if (counter) {
    counter->start();
  }

  cpu_mem_counter_t cpu_counter;
  cpu_counter.start();

  // Run each ep context set
  for (size_t set_index = 0; set_index < set_size; ++set_index) {
    auto ep_context_path = get_ep_context_path(set_index);

    session_options_t session_options{
        device_t(ov_device_name),
        {{kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(ep_context_path)}}};

    sessions.emplace_back(get_model_info().create_session(std::move(session_options)));
    sessions.back()->run_and_validate();
  }

  if (counter) {
    auto delta = counter->delta_from_start();
    EXPECT_LT(
        delta.shared_mem_usage_,
        memory_utils::byte_to_mb(get_external_weights_size() * (set_size / 2)))
        << set_size << " sessions " << delta;
  }

  auto delta = cpu_counter.delta_from_start();
  EXPECT_LT(
      delta.resident_shared_memory_,
      memory_utils::byte_to_mb(get_external_weights_size() * (set_size / 2)))
      << set_size << " sessions " << delta;
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_shared_external_initializer,
    device_parameterized,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<device_parameterized::ParamType>& info) { return info.param; });

#endif  // _WIN32
