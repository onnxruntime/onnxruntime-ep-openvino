// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"

#include <fstream>

using namespace ovep_tests;

namespace {

bool has_ovep_profiler_event(const fs::path& profile_path) {
  std::ifstream file(profile_path);
  if (!file.is_open()) {
    OVEP_TESTS_LOG_DEBUG << "Profile file could not be opened: " << profile_path;
    return false;
  }

  auto json = nlohmann::json::parse(file, nullptr, false);
  if (json.is_discarded() || !json.is_array()) {
    OVEP_TESTS_LOG_DEBUG << "Profile file is not valid JSON array: " << profile_path;
    return false;
  }

  for (const auto& event : json) {
    if (!event.contains("args") || !event["args"].is_object()) {
      continue;
    }

    const auto& args = event["args"];
    if (args.value("ep", "") != "OpenVINOExecutionProvider") {
      continue;
    }

    // ov_exec_type is emitted only by OVEpProfilerImpl when CreateProfiler is used.
    if (args.contains("ov_exec_type") && args["ov_exec_type"].is_string()) {
      return true;
    }
  }

  return false;
}

}  // namespace

struct ort_abi_create_profiler : public ortAbiSingleDeviceTestFixture {};

TEST_P(ort_abi_create_profiler, profiling_emits_ovep_events) {
  const auto& model_info = model_library_t::GetModel("mul_1");
  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);

  auto profile_prefix = temp_dir_ / "create_profiler_profile";
  session_options_wrapper_t session_options(device_wrapper_t(GetParam()), network);
  session_options.EnableProfiling(profile_prefix.native().c_str());

  auto session = model_info.create_session(std::move(session_options));
  session->run_and_validate();

  Ort::AllocatorWithDefaultOptions allocator;
  auto profile_file = session->ort_session().EndProfilingAllocated(allocator);
  const fs::path profile_path(profile_file.get());

  ASSERT_TRUE(fs::exists(profile_path)) << "Profile file not found: " << profile_path;
  EXPECT_TRUE(has_ovep_profiler_event(profile_path))
      << "Expected OV EP profiler events in ORT profile output";

  fs::remove(profile_path);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_create_profiler,
    ort_abi_create_profiler,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_create_profiler::ParamType>& info) { return info.param; });
