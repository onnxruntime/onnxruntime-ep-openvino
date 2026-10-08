// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ort_helper.h"
#include "test_paths.h"

///////////////////////////////////////////////////////////////////////
struct ort_abi_offline_compile_model
    : public ::testing::Test,
      public testing::WithParamInterface<std::tuple<OrtHardwareDeviceType, uint32_t, embed_mode_t> > {
  static constexpr const char* registration_name_ = "OpenVINOExecutionProvider.virtual";

#ifdef WIN32
  const std::filesystem::path default_abi_ep_library_path_ = "onnxruntime_providers_openvino_plugin.dll";
#else
  const std::filesystem::path default_abi_ep_library_path_ = "libonnxruntime_providers_openvino_plugin.so";
#endif

  ///////////////////////////////////////////////////////////////////////
  void SetUp() override {
    fs::remove_all(ovep_tests::GetProcessTempDir());
    fs::create_directories(ovep_tests::GetProcessTempDir());

    env_.RegisterExecutionProviderLibrary(registration_name_, default_abi_ep_library_path_.native().c_str());
  }

  ///////////////////////////////////////////////////////////////////////
  void TearDown() override {
    env_.UnregisterExecutionProviderLibrary(registration_name_);
    fs::remove_all(ovep_tests::GetProcessTempDir());
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_offline_compile_model, run) {
  const auto& device_type = std::get<0>(GetParam());
  const auto& device_id = std::get<1>(GetParam());
  const auto& embed_mode = std::get<2>(GetParam());

  std::vector<Ort::ConstEpDevice> ep_devices;
  for (const auto& device : env_.GetEpDevices()) {
    if (std::strcmp(device.EpName(), "OpenVINOExecutionProvider") != 0) {
      continue;
    }

    const auto& device_metadata_kv_pairs = device.Device().Metadata().GetKeyValuePairs();
    if (device_metadata_kv_pairs.find(kOrtHardwareDevice_MetadataKey_IsVirtual) ==
            device_metadata_kv_pairs.end() ||
        device_metadata_kv_pairs.at(kOrtHardwareDevice_MetadataKey_IsVirtual) != "1") {
      continue;
    }

    if (device.Device().Type() == device_type && device.Device().DeviceId() == device_id) {
      ep_devices.push_back(device);
    }
  }

  EXPECT_EQ(ep_devices.size(), 1) << " Expected to find exactly one matching EP device";

  const auto input_model_path = fs::path("testdata") / "mul_1.onnx";
  const auto output_model_path = ovep_tests::GetProcessTempDir() / "mul_1.onnx";

  Ort::SessionOptions session_options{};
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(env_, ep_devices, ep_options);

  Ort::ModelCompilationOptions compile_options(env_, session_options);
  compile_options.SetEpContextEmbedMode(embed_mode == embed_mode_t::on);
  compile_options.SetInputModelPath(input_model_path.native().c_str());
  compile_options.SetOutputModelPath(output_model_path.native().c_str());

  Ort::CompileModel(env_, compile_options);

  EXPECT_TRUE(fs::exists(output_model_path));

  if (embed_mode == embed_mode_t::off) {
    auto blob_found = [&]() {
      for (const auto& itr : fs::directory_iterator(ovep_tests::GetProcessTempDir())) {
        if (fs::is_regular_file(itr.path()) && itr.path().extension().string() == ".bin") {
          return true;
        }
      }

      return false;
    };

    EXPECT_TRUE(blob_found());
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_offline_model_compilation,
    ort_abi_offline_compile_model,
    testing::Combine(
        testing::Values(OrtHardwareDeviceType_NPU),
        testing::Values(0x643E, 0xB03E, 0xFD3E, 0xD71D),
        testing::Values(embed_mode_t::on, embed_mode_t::off)),
    [](const testing::TestParamInfo<ort_abi_offline_compile_model::ParamType>& info) {
      return "device_id_" + std::to_string(std::get<1>(info.param)) + "_embed_" +
             to_string(std::get<2>(info.param));
    });
