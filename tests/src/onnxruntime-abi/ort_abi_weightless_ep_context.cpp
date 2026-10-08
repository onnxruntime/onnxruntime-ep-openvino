// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ortAbiTestFixture.h"

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
enum class weights_location_t { co_located,
                                source_model_path,
                                source_model_buffer,
                                external_initializers_folder,
                                source_model_path_and_folder,
                                epctx_from_memory };

static std::string to_string(weights_location_t location) {
  switch (location) {
    case weights_location_t::co_located:
      return "co_located";
    case weights_location_t::source_model_path:
      return "source_model_path";
    case weights_location_t::source_model_buffer:
      return "source_model_buffer";
    case weights_location_t::external_initializers_folder:
      return "external_initializers_folder";
    case weights_location_t::source_model_path_and_folder:
      return "source_model_path_and_folder";
    case weights_location_t::epctx_from_memory:
      return "epctx_from_memory";
  }
  return "unknown";
}

///////////////////////////////////////////////////////////////////////
struct ort_abi_weightless_ep_context
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, weights_location_t> > {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_weightless_ep_context, generate_and_run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto weights_location = std::get<1>(GetParam());
  const bool co_located = weights_location == weights_location_t::co_located;
  const auto& model_info = model_library_t::GetModel("gemm_external");

  const fs::path source_dir = temp_dir_ / "src";
  const fs::path ctx_dir = co_located ? source_dir : temp_dir_ / "ctx";
  fs::create_directories(source_dir);
  fs::create_directories(ctx_dir);

  const fs::path model_path = source_dir / model_info.model_path_.filename();
  fs::copy_file(model_info.model_path_, model_path, fs::copy_options::overwrite_existing);
  fs::copy_file(
      model_info.external_data_path_,
      source_dir / model_info.external_data_path_.filename(),
      fs::copy_options::overwrite_existing);

  const fs::path weights_path = source_dir / model_info.external_data_path_.filename();
  const fs::path epctx_model = ctx_dir / "gemm_external_weightless_ctx.onnx";

  {
    network_t network = network_t::create(model_path, session_model_load_mode_t::file);
    network.ort_session_config_.emplace(kOrtSessionOptionEpContextEnable, "1");
    network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
    network.ort_session_config_.emplace(kOrtSessionOptionEpEnableWeightless, "1");
    model_info.create_session(ov_device_name, std::move(network));
    ASSERT_TRUE(fs::exists(epctx_model));
  }

  // Model the app holds in memory because it is encrypted on disk
  std::vector<uint8_t> source_model_bytes;
  if (weights_location == weights_location_t::source_model_buffer) {
    data_utils::read_binary_file(model_path, source_model_bytes);
  }

  if (weights_location == weights_location_t::external_initializers_folder ||
      weights_location == weights_location_t::source_model_buffer ||
      weights_location == weights_location_t::epctx_from_memory) {
    fs::remove(model_path);
  }
  ASSERT_TRUE(fs::exists(weights_path));

  {
    const auto load_mode = weights_location == weights_location_t::epctx_from_memory
                               ? session_model_load_mode_t::memory
                               : session_model_load_mode_t::file;
    network_t network = network_t::create(epctx_model, load_mode);
    switch (weights_location) {
      case weights_location_t::co_located:
        // Nothing configured: the weights are expected beside the EPContext model. The source model
        // is never consulted, so only ov::weights_path is set.
        break;
      case weights_location_t::source_model_path:
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextSourceModelPath, path_to_utf8(model_path));
        break;
      case weights_location_t::source_model_buffer:
        // A buffer carries no directory of its own, so the weights folder
        // has to be named for the model's external initializers to resolve.
        network.weightless_source_model_buffer_ = source_model_bytes;
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath, path_to_utf8(source_dir));
        break;
      case weights_location_t::external_initializers_folder:
        // Using this option the EP has to be told where the bin is located.
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath, path_to_utf8(source_dir));
        break;
      case weights_location_t::source_model_path_and_folder:
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextSourceModelPath, path_to_utf8(model_path));
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath, path_to_utf8(source_dir));
        break;
      case weights_location_t::epctx_from_memory:
        // Loaded from bytes, so ORT reports no model path and the EP has to be told where the
        // EPContext model lives to find the bin beside it
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath, path_to_utf8(source_dir));
        break;
    }
    auto session = model_info.create_session(ov_device_name, std::move(network));
    session->run_and_validate();
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_weightless_ep_context,
    testing::Combine(
        testing::Values(std::string("NPU")),
        testing::Values(weights_location_t::co_located,
                        weights_location_t::source_model_path,
                        weights_location_t::source_model_buffer,
                        weights_location_t::external_initializers_folder,
                        weights_location_t::source_model_path_and_folder,
                        weights_location_t::epctx_from_memory)),
    [](const testing::TestParamInfo<ort_abi_weightless_ep_context::ParamType>& info) {
      return std::get<0>(info.param) + "_" + to_string(std::get<1>(info.param));
    });

///////////////////////////////////////////////////////////////////////////
// ep.context_source_model_path names the source model, so pointing it at the weights is rejected
// rather than silently handed to the ONNX frontend as a model.
TEST_F(ortAbiTestFixture, ort_abi_weightless_source_model_path_rejects_non_onnx) {
  const auto& model_info = model_library_t::GetModel("gemm_external");

  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);
  network.ort_session_config_.emplace(
      kOrtSessionOptionEpContextSourceModelPath, path_to_utf8(model_info.external_data_path_));

  EXPECT_THROW({ auto session = model_info.create_session("CPU", std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////
struct ort_abi_weightless_ep_context_internal : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
// Weightless on an internal-only model (no external weights) must be rejected.
TEST_P(ort_abi_weightless_ep_context_internal, internal_only_model_rejected) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("mul_1");

  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);
  network.ort_session_config_.emplace(kOrtSessionOptionEpContextEnable, "1");
  network.ort_session_config_.emplace(
      kOrtSessionOptionEpContextFilePath, path_to_utf8(temp_dir_ / "mul_1_weightless_ctx.onnx"));
  network.ort_session_config_.emplace(kOrtSessionOptionEpEnableWeightless, "1");

  EXPECT_THROW({ auto session = model_info.create_session(ov_device_name, std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_weightless_ep_context_internal,
    testing::Values(std::string("NPU")),
    [](const testing::TestParamInfo<ort_abi_weightless_ep_context_internal::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_weightless_ep_context_unresolved
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, session_model_load_mode_t> > {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
// The EPContext model is generated away from the weights and the source model is then removed. With
// no session options the EP can only look beside the EPContext model, so neither the weights nor the
// blob are reachable.
TEST_P(ort_abi_weightless_ep_context_unresolved, no_session_options_rejected) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto load_mode = std::get<1>(GetParam());
  const auto& model_info = model_library_t::GetModel("gemm_external");

  const fs::path source_dir = temp_dir_ / "src";
  const fs::path ctx_dir = temp_dir_ / "ctx";
  fs::create_directories(source_dir);
  fs::create_directories(ctx_dir);

  const fs::path model_path = source_dir / model_info.model_path_.filename();
  fs::copy_file(model_info.model_path_, model_path, fs::copy_options::overwrite_existing);
  fs::copy_file(
      model_info.external_data_path_,
      source_dir / model_info.external_data_path_.filename(),
      fs::copy_options::overwrite_existing);

  const fs::path epctx_model = ctx_dir / "gemm_external_weightless_ctx.onnx";

  {
    network_t network = network_t::create(model_path, session_model_load_mode_t::file);
    network.ort_session_config_.emplace(kOrtSessionOptionEpContextEnable, "1");
    network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model));
    network.ort_session_config_.emplace(kOrtSessionOptionEpEnableWeightless, "1");
    model_info.create_session(ov_device_name, std::move(network));
    ASSERT_TRUE(fs::exists(epctx_model));
  }

  fs::remove(model_path);

  network_t network = network_t::create(epctx_model, load_mode);
  EXPECT_THROW({ auto session = model_info.create_session(ov_device_name, std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_weightless_ep_context_unresolved,
    testing::Combine(
        testing::Values(std::string("NPU")),
        testing::Values(session_model_load_mode_t::file, session_model_load_mode_t::memory)),
    [](const testing::TestParamInfo<ort_abi_weightless_ep_context_unresolved::ParamType>& info) {
      return std::get<0>(info.param) + (std::get<1>(info.param) == session_model_load_mode_t::file
                                            ? "_epctx_from_file"
                                            : "_epctx_from_memory");
    });

///////////////////////////////////////////////////////////////////////
struct ort_abi_weightless_ep_context_shared : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
// GetWeightlessSupport reports NONE while sharing is on, so ORT rejects the session up front.
TEST_P(ort_abi_weightless_ep_context_shared, weightless_with_sharing_rejected) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("gemm_external");

  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);
  network.ort_session_config_.emplace(kOrtSessionOptionEpContextEnable, "1");
  network.ort_session_config_.emplace(kOrtSessionOptionEpContextEmbedMode, "0");
  network.ort_session_config_.emplace(
      kOrtSessionOptionEpContextFilePath, path_to_utf8(temp_dir_ / "gemm_external_shared_weightless_ctx.onnx"));
  network.ort_session_config_.emplace(kOrtSessionOptionEpEnableWeightless, "1");
  network.ort_session_config_.emplace(kOrtSessionOptionShareEpContexts, "1");

  EXPECT_THROW({ auto session = model_info.create_session(ov_device_name, std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_weightless_ep_context_shared,
    testing::Values(std::string("NPU")),
    [](const testing::TestParamInfo<ort_abi_weightless_ep_context_shared::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_weightless_ep_context_unsupported_device : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
// Only NPU blobs can be weightless, so GetWeightlessSupport reports NONE elsewhere and ORT
// rejects the session rather than letting the EP accept the option.
TEST_P(ort_abi_weightless_ep_context_unsupported_device, weightless_rejected) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("gemm_external");

  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);
  network.ort_session_config_.emplace(kOrtSessionOptionEpEnableWeightless, "1");

  EXPECT_THROW({ auto session = model_info.create_session(ov_device_name, std::move(network)); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_weightless_ep_context_unsupported_device,
    testing::Values(std::string("CPU"), std::string("GPU")),
    [](const testing::TestParamInfo<ort_abi_weightless_ep_context_unsupported_device::ParamType>& info) {
      return info.param;
    });
