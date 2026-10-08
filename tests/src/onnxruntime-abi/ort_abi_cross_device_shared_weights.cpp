// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// This file is Windows-only due to platform-specific memory counter functionality
#ifdef _WIN32

#include "ortAbiTestFixture.h"
#include "memory.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;

// Verifies cross-device shared weights: when a GPU session and an NPU session in the same process
// reference the same external weights file, the NPU imports the GPU's host allocation (Level-Zero
// CPU_VA) instead of mapping a second copy. Both EPContext models are generated from the same source
// model, so their serialized weight `location` resolves to the identical absolute path, which is what
// enables sharing via WeightFileManager. Ordering matters: the GPU session must load first because the
// NPU import path only triggers after a GPU mapping already exists.
struct ort_abi_cross_device_shared_weights_fixture : public ortAbiTestFixture {
  std::vector<std::string> additional_required_devices() const override { return {"GPU", "NPU"}; }

  const auto& get_model_info() { return model_library_t::GetModel("gemm_external"); }

  fs::path get_external_data_file() { return get_model_info().external_data_path_; }
  auto get_external_weights_size() { return fs::file_size(get_external_data_file()); }

  // Generate a self-contained shared EPContext model for `device` at `ctx_path`.
  void generate_ep_context(const std::string& device, const fs::path& ctx_path) {
    session_options_t session_options{
        device_t(device),
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionStopShareEpContexts, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(ctx_path)}}};

    get_model_info().create_session(std::move(session_options));
    ASSERT_TRUE(fs::exists(ctx_path));
  }

  // Generate GPU and NPU EPContext models that point at one shared provider binary. The first
  // session contributes the GPU native blob and leaves the active shared context open; the second
  // contributes the NPU native blob and finalizes the shared binary. Distinct output paths preserve
  // the native partition names while the active shared context preserves the common ep_cache_context path.
  void generate_common_ep_context(const fs::path& gpu_ctx_path, const fs::path& npu_ctx_path) {
    session_options_t gpu_options{
        device_t("GPU"),
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionStopShareEpContexts, "0"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(gpu_ctx_path)}}};

    // Keep the first session alive while the second session is created so the active shared
    // context remains available for both generation passes.
    auto gpu_session = get_model_info().create_session(std::move(gpu_options));
    ASSERT_TRUE(fs::exists(gpu_ctx_path));

    session_options_t npu_options{
        device_t("NPU"),
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionStopShareEpContexts, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(npu_ctx_path)}}};

    auto npu_session = get_model_info().create_session(std::move(npu_options));
    ASSERT_TRUE(fs::exists(npu_ctx_path));
  }

  // Load a previously generated shared EPContext model for `device`.
  std::unique_ptr<builtin_session_t> load_ep_context(const std::string& device, const fs::path& ctx_path) {
    session_options_t session_options{
        device_t(device),
        {{kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(ctx_path)}}};

    return get_model_info().create_session(std::move(session_options));
  }

  std::unique_ptr<builtin_session_t> warm_up_device(const std::string& device) {
    try {
      auto session = model_library_t::GetModel("mul_1").create_session(session_options_t{device_t(device)});
      session->run_and_validate();
      return session;
    } catch (const std::exception& e) {
      OVEP_TESTS_LOG_WARN << device << " warm-up failed (continuing): " << e.what();
      return nullptr;
    }
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_cross_device_shared_weights_fixture, gpu_npu_share_external_weights) {
  device_t gpu("GPU");
  device_t npu("NPU");

  const auto gpu_ctx = temp_dir_ / "gemm_external_ctx_GPU.onnx";
  const auto npu_ctx = temp_dir_ / "gemm_external_ctx_NPU.onnx";

  // Generate one shared EPContext model per device from the same source model. Both reference the
  // same source-model external weights file, so the weights can be shared across devices at load.
  generate_ep_context("GPU", gpu_ctx);
  generate_ep_context("NPU", npu_ctx);

  std::vector<std::unique_ptr<builtin_session_t>> sessions;

  // Warm up the NPU runtime on an unrelated model so its driver DLL images and device-init memory are
  // already resident and not attributed to the measured NPU increment below. Best-effort.
  try {
    auto warmup = model_library_t::GetModel("mul_1").create_session(session_options_t{device_t("NPU")});
    warmup->run_and_validate();
  } catch (const std::exception& e) {
    OVEP_TESTS_LOG_WARN << "NPU warm-up failed (continuing): " << e.what();
  }

  // GPU loads first and allocates the weights (iGPU host tensor). This must precede the NPU session so
  // the NPU import path finds the existing "GPU.0" mapping to reuse.
  sessions.emplace_back(load_ep_context("GPU", gpu_ctx));
  sessions.back()->run_and_validate();

  // Measure only the memory added by the NPU session.
  cpu_mem_counter_t cpu_counter;
  cpu_counter.start();

  sessions.emplace_back(load_ep_context("NPU", npu_ctx));
  sessions.back()->run_and_validate();

  const auto delta = cpu_counter.delta_from_start();
  OVEP_TESTS_LOG_INFO << "NPU session memory increment:\n"
                      << delta;

  // If the NPU imported the GPU allocation, it adds ~0 weight memory; without sharing it would add a
  // full second copy of the weights (~weights_size).
  EXPECT_LT(delta.resident_shared_memory_,
            memory_utils::byte_to_mb(get_external_weights_size()))
      << "NPU appears to have allocated its own copy of the weights.\n"
      << delta;
}

TEST_F(ort_abi_cross_device_shared_weights_fixture, gpu_npu_share_weights_with_common_ep_context) {
  device_t gpu("GPU");
  device_t npu("NPU");

  const auto gpu_ctx = temp_dir_ / "gemm_external_ctx_shared_GPU.onnx";
  const auto npu_ctx = temp_dir_ / "gemm_external_ctx_shared_NPU.onnx";
  generate_common_ep_context(gpu_ctx, npu_ctx);

  // Keep the NPU runtime warm so its process/device initialization is not attributed to weight
  // sharing. The GPU session itself is created before the measurement starts.
  auto npu_warmup_session = warm_up_device("NPU");

  std::vector<std::unique_ptr<builtin_session_t>> sessions;

  sessions.emplace_back(load_ep_context("GPU", gpu_ctx));
  // Run GPU session to fault resident weights before counter starts (important when gpu uses cpu_va).
  // Otherwise NPU load is first to touch and mmap is charged to NPU, even though genuinely shared.
  sessions.back()->run_and_validate();

  cpu_mem_counter_t cpu_counter;
  cpu_counter.start();
  sessions.emplace_back(load_ep_context("NPU", npu_ctx));
  const auto delta = cpu_counter.delta_from_start();

  OVEP_TESTS_LOG_INFO << "NPU session memory increment with a common EPContext:\n"
                      << delta;

  // Shared weights are bound while the initial infer request is created during session setup, so
  // measure only NPU context creation. First-run driver allocations are intentionally excluded.
  EXPECT_LT(delta.resident_shared_memory_,
            memory_utils::byte_to_mb(get_external_weights_size()))
      << "NPU appears to have allocated its own copy of the shared weights when using a common EPContext.\n"
      << delta;
}

#endif  // _WIN32
