// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// Verifies ep.openvinoexecutionprovider.hardware_device_type, an env-level config entry that can
// only be set on the first OrtEnv created in a process (see main.cpp's MakeEnv()) -- so this needs
// a fresh process per filter value. EXPECT_EXIT provides that, forced to "threadsafe" style: the
// POSIX default ("fast", fork-only) resumes inside the already-initialized parent process instead
// of re-running main(), so the filter would never actually reach env_.

#include "ortAbiTestFixture.h"

#include <gtest/gtest.h>

#include <set>
#include <sstream>

///////////////////////////////////////////////////////////////////////////
using namespace ovep_tests;

namespace {

constexpr const char* kHardwareDeviceTypeFilterEnvVar = "OVEP_TESTS_HARDWARE_DEVICE_TYPE_FILTER";
constexpr const char* kVirtualDevicesEnvVar = "OVEP_TESTS_ALLOW_VIRTUAL_DEVICES";
constexpr const char* kInvalidFilterValue = "BOGUS";

// Splits "CPU;GPU" into {"CPU", "GPU"}, skipping empty segments.
std::vector<std::string> SplitFilterValue(std::string_view value) {
  std::vector<std::string> tokens;
  for (auto&& token : std::ranges::split_view(value, ';')) {
    std::string token_str(token.begin(), token.end());
    if (!token_str.empty()) {
      tokens.push_back(std::move(token_str));
    }
  }
  return tokens;
}

}  // namespace

///////////////////////////////////////////////////////////////////////////
// Not ortAbiTestFixture's SetUp(): it unconditionally requires CPU, which a CPU-excluding filter
// would exclude, and SetUp() reruns inside the death-test child too (with the filter active).
struct ort_abi_hardware_device_type_filter : public ortAbiTestFixture,
                                             public ::testing::WithParamInterface<std::string> {
  void SetUp() override {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    RegisterPluginProvider(GetPluginProviderPath());

    const auto required = SplitFilterValue(GetParam());
    const auto unavailable = get_unavailable_openvino_devices(env_.GetEpDevices(), required);
    if (unavailable.empty()) {
      return;
    }

    std::ostringstream message;
    message << "Unavailable OpenVINO device(s) for filter '" << GetParam() << "': ";
    for (size_t i = 0; i < unavailable.size(); ++i) {
      if (i != 0) {
        message << ", ";
      }
      message << unavailable[i];
    }

    if (skip_unavailable_tests_enabled()) {
      GTEST_SKIP() << message.str();
    }
    FAIL() << message.str() << ". Set " << kSkipUnavailableTestsEnvironmentVariable << "=1 to skip this test.";
  }

  void TearDown() override { DeregisterPluginProvider(); }
};

TEST_P(ort_abi_hardware_device_type_filter, enumerates_only_configured_types) {
  scoped_environment_var filter_var(kHardwareDeviceTypeFilterEnvVar, GetParam());
  const auto expected_types = SplitFilterValue(GetParam());

  EXPECT_EXIT(
      {
        // Tracked separately from the AUTO meta-device factory's own EpDevices (EpName() ==
        // "OpenVINOExecutionProvider.AUTO"), which are filtered via the same MatchOvDevice() path
        // as the plain physical-device factory but are otherwise independent entries.
        std::set<std::string> seen_types;
        std::set<std::string> seen_auto_types;
        bool all_matched = true;
        for (const auto& device : env_.GetEpDevices()) {
          std::string_view ep_name = device.EpName();
          if (!ep_name.starts_with("OpenVINOExecutionProvider")) {
            continue;
          }
          std::string ov_device = device.EpMetadata().GetValue("ov_device");
          auto matched = std::ranges::find_if(
              expected_types, [&](const std::string& type) { return ov_device.starts_with(type); });
          if (matched == expected_types.end()) {
            all_matched = false;
            continue;
          }
          seen_types.insert(*matched);
          if (ep_name == "OpenVINOExecutionProvider.AUTO") {
            seen_auto_types.insert(*matched);
          }
        }
        for (const auto& type : expected_types) {
          if (!seen_types.contains(type) || !seen_auto_types.contains(type)) {
            all_matched = false;
          }
        }
        std::exit(all_matched ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    device_type_combinations, ort_abi_hardware_device_type_filter,
    ::testing::Values("CPU", "GPU", "NPU", "CPU;GPU", "CPU;;GPU;", "GPU;NPU", "CPU;NPU"),
    [](const ::testing::TestParamInfo<std::string>& info) {
      std::string name = info.param;
      std::ranges::replace(name, ';', '_');
      return name;
    });

///////////////////////////////////////////////////////////////////////////
// No fixture: registration is expected to throw here, so it must happen inside the EXPECT_EXIT
// statement rather than a SetUp() that would throw uncaught before the child reaches it.
TEST(ort_abi_hardware_device_type_filter_invalid_value, rejects_invalid_filter_value) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  scoped_environment_var filter_var(kHardwareDeviceTypeFilterEnvVar, kInvalidFilterValue);

  EXPECT_EXIT(
      {
        bool threw = false;
        try {
          env_.RegisterExecutionProviderLibrary("OpenVINOExecutionProvider",
                                                ortAbiTestFixture::GetPluginProviderPath().native().c_str());
        } catch (const Ort::Exception&) {
          threw = true;
        }
        std::exit(threw ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

///////////////////////////////////////////////////////////////////////////
// GetSupportedVirtualDevices() synthesizes NPU devices from a hardcoded platform-ID list, so this
// exercises the filter's virtual-device guard without needing real NPU hardware.
TEST(ort_abi_hardware_device_type_filter_virtual, enumerates_virtual_devices_when_npu_included) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  scoped_environment_var virtual_var(kVirtualDevicesEnvVar, "1");
  scoped_environment_var filter_var(kHardwareDeviceTypeFilterEnvVar, "NPU");

  EXPECT_EXIT(
      {
        env_.RegisterExecutionProviderLibrary("OpenVINOExecutionProvider",
                                              ortAbiTestFixture::GetPluginProviderPath().native().c_str());
        bool saw_npu = false;
        for (const auto& device : env_.GetEpDevices()) {
          std::string_view ep_name = device.EpName();
          if (!ep_name.starts_with("OpenVINOExecutionProvider")) {
            continue;
          }
          std::string ov_device = device.EpMetadata().GetValue("ov_device");
          if (ov_device.starts_with("NPU")) {
            saw_npu = true;
          }
        }
        std::exit(saw_npu ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(ort_abi_hardware_device_type_filter_virtual, excludes_virtual_devices_when_npu_filtered_out) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  scoped_environment_var virtual_var(kVirtualDevicesEnvVar, "1");
  scoped_environment_var filter_var(kHardwareDeviceTypeFilterEnvVar, "CPU");

  EXPECT_EXIT(
      {
        env_.RegisterExecutionProviderLibrary("OpenVINOExecutionProvider",
                                              ortAbiTestFixture::GetPluginProviderPath().native().c_str());
        bool saw_any = false;
        for (const auto& device : env_.GetEpDevices()) {
          std::string_view ep_name = device.EpName();
          saw_any = saw_any || ep_name.starts_with("OpenVINOExecutionProvider");
        }
        std::exit(saw_any ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}
