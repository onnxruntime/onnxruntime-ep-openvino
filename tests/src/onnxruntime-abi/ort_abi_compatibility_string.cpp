// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_compatibility_string : public ortAbiSingleDeviceTestFixture {};

TEST_P(ort_abi_compatibility_string, run) {
  const auto& ov_device_name = GetParam();

  device_t device(ov_device_name);

  const auto& model_info = model_library_t::GetModel("mul_1");
  fs::path epctx_model = temp_dir_ / ("mul_1_compat_" + ov_device_name + ".onnx");

  // Generate EPContext model
  {
    session_options_wrapper_t session_options(
        ov_device_name,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
         {kOrtSessionOptionEpContextEmbedMode, "1"}});

    auto session = model_info.create_session(std::move(session_options));
    ASSERT_TRUE(fs::exists(epctx_model)) << "EPContext model was not generated";
  }

  // Extract compatibility string
  auto compat_info = model_library_t::model_info_t::GetCompatibilityString(epctx_model);

  // Build a fake JSON compat string for validation tests
  nlohmann::ordered_json fake_j;
  fake_j["ovep_compat_ver"] = "1.0";
  fake_j["ovep_ver"] = "0.0.0";
  fake_j["ovep_bin_ver"] = "1.1";
  fake_j["ov_compat_string"] = "meta=0.0;ov=1999.0.0;desc=[compiler=0.0;npu=9999;t=0;elf=0.0.0;mi=0.0.0]";
  const std::string fake_compat_string = fake_j.dump();

  // EP always produces JSON regardless of device
  ASSERT_NE(compat_info.get(), nullptr);
  ASSERT_GT(std::strlen(compat_info.get()), 0u);

  // Verify JSON envelope structure
  auto j = nlohmann::json::parse(compat_info.get());
  EXPECT_TRUE(j.contains("ovep_compat_ver"));
  EXPECT_TRUE(j.contains("ovep_ver"));
  EXPECT_TRUE(j.contains("ovep_bin_ver"));
  EXPECT_TRUE(j.contains("ov_compat_string"));

  OVEP_TESTS_LOG_INFO << ov_device_name << " compatibility string: " << compat_info.get();

  // Devices whose OpenVINO plugin implements ov::runtime_requirements /
  // ov::compatibility_check. These emit a non-empty descriptor and can answer
  // SUPPORTED/UNSUPPORTED for it.
  //
  // All supported OpenVINO devices emit a descriptor and can validate it.
  constexpr bool device_validates_ov_string = true;

  if (device_validates_ov_string && j["ov_compat_string"].get<std::string>().empty()) {
    FAIL() << "OpenVINO device '" << ov_device_name
           << "' does not provide a compatibility string; compatibility validation is not applicable.";
  }

  if (device_validates_ov_string) {
    EXPECT_FALSE(j["ov_compat_string"].get<std::string>().empty());

    // Valid string from the same device -> SUPPORTED
    {
      auto result = device.ValidateCompatibility(compat_info.get());
      EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL)
          << "Model compiled on current " << ov_device_name << " should be compatible";
    }

    // Tampered descriptor (valid envelope, mutated ov_compat_string) -> UNSUPPORTED.
    // The descriptor is matched by exact equality, so any mutation invalidates it.
    {
      auto tampered_j = nlohmann::json::parse(compat_info.get());
      tampered_j["ov_compat_string"] =
          tampered_j["ov_compat_string"].get<std::string>() + ";tampered=1";
      const std::string tampered_string = tampered_j.dump();

      auto result = device.ValidateCompatibility(tampered_string.c_str());
      EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_UNSUPPORTED)
          << "Tampered OV string should be unsupported on " << ov_device_name;
    }

    // Foreign descriptor from a different device/generation -> UNSUPPORTED
    {
      auto result = device.ValidateCompatibility(fake_compat_string.c_str());
      EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_UNSUPPORTED)
          << "Foreign OV string should be unsupported on " << ov_device_name;
    }
  }

  // ---- Device-agnostic envelope (bin version) gating ----

  // Newer MAJOR bin version -> UNSUPPORTED (gated). Empty OV string keeps the check
  // focused on the bin-version gate alone.
  {
    nlohmann::ordered_json new_bin_j;
    new_bin_j["ovep_compat_ver"] = "1.0";
    new_bin_j["ovep_ver"] = "0.0.0";
    new_bin_j["ovep_bin_ver"] = "99.99";
    new_bin_j["ov_compat_string"] = "";
    std::string new_bin_string = new_bin_j.dump();

    auto result = device.ValidateCompatibility(new_bin_string.c_str());
    EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_UNSUPPORTED)
        << ov_device_name << " should reject newer major bin version";
  }

  // Newer MINOR bin version is recorded but not gated -> bin check passes; empty OV
  // string then yields NOT_APPLICABLE on every device.
  {
    nlohmann::ordered_json minor_bump_j;
    minor_bump_j["ovep_compat_ver"] = "1.0";
    minor_bump_j["ovep_ver"] = "0.0.0";
    minor_bump_j["ovep_bin_ver"] = "1.99";
    minor_bump_j["ov_compat_string"] = "";
    std::string minor_bump_string = minor_bump_j.dump();

    auto result = device.ValidateCompatibility(minor_bump_string.c_str());
    EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE)
        << ov_device_name << " should accept a newer minor bin version (minor is not gated)";
  }

  // Empty string -> NOT_APPLICABLE for all devices
  {
    auto result = device.ValidateCompatibility("");
    EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE)
        << "Empty string should return NOT_APPLICABLE";
  }

  // Invalid JSON -> UNSUPPORTED for all devices
  {
    auto result = device.ValidateCompatibility("not-valid-json");
    EXPECT_EQ(result, OrtCompiledModelCompatibility_EP_UNSUPPORTED)
        << "Invalid JSON should return UNSUPPORTED";
  }
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_compatibility_string,
    ort_abi_compatibility_string,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_compatibility_string::ParamType>& info) { return info.param; });
