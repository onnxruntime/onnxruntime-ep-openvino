// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "env.h"
#include "test_availability.h"
#include "ort_abi_device_wrapper.h"
#include "ort_abi_session_options_wrapper.h"
#include "ort_abi_external_resource_importer_wrapper.h"
#include "ort_abi_external_memory_wrapper.h"
#include "ort_session_wrapper.h"
#include "ort_graph_wrapper.h"
#include "ort_memory_info_wrapper.h"
#include "ort_pipeline_wrapper.h"
#include "ort_io_binding_wrapper.h"
#include "ort_builtin_model_library.h"
#include "ort_custom_ops_wrapper.h"
#include "test_paths.h"

using ovep_tests::scoped_environment_var;

///////////////////////////////////////////////////////////////////////////
struct ortAbiTestFixture : public ::testing::Test {
  using fixture_t = ortAbiTestFixture;

  using device_wrapper_t = ovep_tests::onnxruntime_abi::device_wrapper<fixture_t>;
  using session_options_wrapper_t = ovep_tests::onnxruntime_abi::session_options_wrapper<fixture_t>;
  using session_factory_t = ovep_tests::onnxruntime::session_factory<fixture_t>;
  using session_wrapper_t = ovep_tests::onnxruntime::session_wrapper<fixture_t>;
  using graph_wrapper_t = ovep_tests::onnxruntime::graph_wrapper<fixture_t>;
  using memory_info_wrapper_t = ovep_tests::onnxruntime::memory_info_wrapper<fixture_t>;
  using pipeline_wrapper_t = ovep_tests::onnxruntime::pipeline_wrapper<fixture_t>;
  using io_binding_wrapper_t = ovep_tests::onnxruntime::io_binding_wrapper<fixture_t>;
  using model_library_t = ovep_tests::model_library<fixture_t>;
  using builtin_session_t = ovep_tests::builtin_model_session<fixture_t>;
  using custom_ops_wrapper_t = ovep_tests::onnxruntime::custom_ops_wrapper<fixture_t>;
  using external_resource_importer_wrapper_t =
      ovep_tests::onnxruntime_abi::external_resource_importer_wrapper<fixture_t>;
  using external_memory_wrapper_t = ovep_tests::onnxruntime_abi::external_memory_wrapper<fixture_t>;

  static inline const fs::path temp_dir_ = ovep_tests::GetProcessTempDir();
  std::optional<scoped_environment_var> ovep_enable_debug_;
#ifdef _WIN32
  static inline const std::filesystem::path default_abi_ep_library_path =
      L"onnxruntime_providers_openvino_plugin.dll";
  static inline const std::filesystem::path default_legacy_ep_library_path = L"onnxruntime_providers_openvino.dll";
#else
  static inline const std::filesystem::path default_abi_ep_library_path =
      "libonnxruntime_providers_openvino_plugin.so";
  static inline const std::filesystem::path default_legacy_ep_library_path = "libonnxruntime_providers_openvino.so";
#endif

  ep_type_t loaded_ep_ = ep_type_t::abi;

  static fs::path GetPluginProviderPath() {
    auto path = ovep_tests::get_environment_variable("OVEP_TESTS_ORT_EP_LIBRARY_PATH");
    if (path.has_value()) {
      return fs::path(*path);
    }
    return default_abi_ep_library_path;
  }

  static fs::path GetTempDir() { return fs::absolute(temp_dir_); }

  static void ResetTempDir() {
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
  }

  ///////////////////////////////////////////////////////////////////////////
  virtual void SetUp(void) {
    const auto path = GetPluginProviderPath();
    RegisterPluginProvider(path);

    // Delete temp directory used to store test output data
    ResetTempDir();

    std::vector<std::string> required_devices{"CPU"};
    const auto additional_devices = additional_required_devices();
    required_devices.insert(required_devices.end(), additional_devices.begin(), additional_devices.end());

    const auto unavailable_devices =
        ovep_tests::get_unavailable_openvino_devices(env_.GetEpDevices(), required_devices);
    std::vector<std::string> unavailable_resources;
    for (const auto& device : unavailable_devices) {
      unavailable_resources.push_back("OpenVINO device '" + device + "'");
    }
    if (requires_legacy_ep() && !is_legacy_ep_available(path)) {
      unavailable_resources.push_back("legacy OpenVINO execution provider ('" +
                                      default_legacy_ep_library_path.generic_string() + "')");
    }

    if (!unavailable_resources.empty()) {
      std::ostringstream message;
      message << "Unavailable test resource(s): ";
      for (size_t i = 0; i < unavailable_resources.size(); ++i) {
        if (i != 0) {
          message << ", ";
        }
        message << unavailable_resources[i];
      }

      if (ovep_tests::skip_unavailable_tests_enabled()) {
        GTEST_SKIP() << message.str();
      }
      message << ". Set " << ovep_tests::kSkipUnavailableTestsEnvironmentVariable
              << "=1 to skip this test.";
      FAIL() << message.str();
    }
  }

  virtual std::vector<std::string> additional_required_devices() const { return {}; }

  virtual bool requires_legacy_ep() const { return false; }

  bool is_legacy_ep_available(const fs::path& abi_library_path) {
    DeregisterPluginProvider();

    bool available = false;
    try {
      RegisterPluginProvider(default_legacy_ep_library_path);
      available = true;
    } catch (...) {
    }

    if (available) {
      DeregisterPluginProvider();
    }
    RegisterPluginProvider(abi_library_path);
    return available;
  }

  ///////////////////////////////////////////////////////////////////////////
  void DeregisterPluginProvider() { env_.UnregisterExecutionProviderLibrary("OpenVINOExecutionProvider"); }

  ///////////////////////////////////////////////////////////////////////////
  void RegisterPluginProvider(const fs::path& library_path) {
    env_.RegisterExecutionProviderLibrary("OpenVINOExecutionProvider", library_path.native().c_str());

    loaded_ep_ =
        library_path.generic_string().find(default_abi_ep_library_path.generic_string()) != std::string::npos
            ? ep_type_t::abi
            : ep_type_t::legacy;
  }

  ///////////////////////////////////////////////////////////////////////////
  void SetProvider(ep_type_t ep) {
    if (loaded_ep_ == ep) {
      return;
    }

    DeregisterPluginProvider();
    if (ep == ep_type_t::legacy) {
      RegisterPluginProvider(ortAbiTestFixture::default_legacy_ep_library_path);
    } else {
      RegisterPluginProvider(ortAbiTestFixture::default_abi_ep_library_path);
    }

    OVEP_TESTS_LOG_DEBUG << "Switched to " << to_string(ep);
  }

  ///////////////////////////////////////////////////////////////////////////
  static ep_type_t get_opposite_ep(ep_type_t ep) {
    return (ep == ep_type_t::abi) ? ep_type_t::legacy : ep_type_t::abi;
  }

  ///////////////////////////////////////////////////////////////////////////
  void enable_ovep_debug_output() { ovep_enable_debug_.emplace("ORT_OPENVINO_ENABLE_DEBUG", "1"); }

  ///////////////////////////////////////////////////////////////////////////
  void disable_ovep_debug_output() { ovep_enable_debug_.emplace("ORT_OPENVINO_ENABLE_DEBUG", "0"); }

  ///////////////////////////////////////////////////////////////////////////
  virtual void TearDown(void) {
    ovep_enable_debug_.reset();
    DeregisterPluginProvider();
    fs::remove_all(temp_dir_);
  }
};

///////////////////////////////////////////////////////////////////////////
struct ortAbiSingleDeviceTestFixture
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {
  static auto all_devices() {
    return testing::Values(std::string{"CPU"}, std::string{"GPU"}, std::string{"NPU"});
  }

  std::vector<std::string> additional_required_devices() const override { return {GetParam()}; }
};
