// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

#include "ort_test_version.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

struct ort_abi_compile_and_reload_external_epctx
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, weight_sharing_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

TEST_P(ort_abi_compile_and_reload_external_epctx, run) {
  const auto& device = std::get<0>(GetParam());
  const auto weight_sharing = std::get<1>(GetParam());
  const auto& model = model_library_t::GetModel("mul_1");
  const auto context_path = temp_dir_ / "context.onnx";

  // Keep the manager alive after CompileModel releases its temporary session.
  auto keeper = model.create_session(device);
  auto input = model.load_model_buffer();
  session_options_wrapper_t compile_session_options(
      device, {{kOrtSessionOptionEpContextFilePath, path_to_utf8(context_path)},
               {kOrtSessionOptionShareEpContexts, to_string(weight_sharing)},
               {kOrtSessionOptionStopShareEpContexts, to_string(weight_sharing)}});
  Ort::ModelCompilationOptions compile_options(env_, compile_session_options);
  compile_options.SetInputModelFromBuffer(input.data(), input.size());
  compile_options.SetOutputModelPath(context_path.c_str());
  compile_options.SetEpContextEmbedMode(false);
  auto status = Ort::CompileModel(env_, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(fs::exists(context_path));

  auto network = network_t::create(context_path, session_model_load_mode_t::memory);
  network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, path_to_utf8(context_path));
  auto loaded = model.create_session(device, std::move(network));
  loaded->run_and_validate();
  keeper->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context, ort_abi_compile_and_reload_external_epctx,
    testing::Combine(ortAbiSingleDeviceTestFixture::all_devices(),
                     testing::Values(weight_sharing_t::on, weight_sharing_t::off)),
    [](const testing::TestParamInfo<ort_abi_compile_and_reload_external_epctx::ParamType>& info) {
      return std::get<0>(info.param) + "_weight_share_" + to_string(std::get<1>(info.param));
    });

///////////////////////////////////////////////////////////////////////
// Base test implementation that can be shared between fixtures
struct ort_abi_generate_and_run_epctx_model_base : public ortAbiTestFixture {
  void
  run_test(const std::string& ov_device_name, embed_mode_t embed_mode, ep_type_t generator_ep, ep_type_t runner_ep) {
    const auto& model_info = model_library_t::GetModel("mul_1");
    fs::path epctx_model = "mul_1_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx";
    epctx_model = temp_dir_ / epctx_model;

    // Generate epctx model
    {
      SetProvider(generator_ep);

      session_options_wrapper_t session_options(
          ov_device_name,
          {{kOrtSessionOptionEpContextEnable, "1"},
           {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
           {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)}});

      auto session = model_info.create_session(std::move(session_options));

      EXPECT_TRUE(fs::exists(epctx_model));
    }

    // Run epctx model
    {
      SetProvider(runner_ep);

      session_options_wrapper_t session_options(
          ov_device_name,
          {{kOrtSessionOptionEpContextFilePath, epctx_model.string()}});

      auto session = model_info.create_session(std::move(session_options));
      session->run_and_validate();
    }
  }
};

///////////////////////////////////////////////////////////////////////
struct ort_abi_generate_and_run_epctx_model
    : public ort_abi_generate_and_run_epctx_model_base,
      public testing::WithParamInterface<std::tuple<std::string, embed_mode_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_generate_and_run_epctx_model, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());

  run_test(ov_device_name, embed_mode, ep_type_t::abi, ep_type_t::abi);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_generate_and_run_epctx_model,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off)),
    [](const testing::TestParamInfo<ort_abi_generate_and_run_epctx_model::ParamType>& info) { return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)); });

///////////////////////////////////////////////////////////////////////////
struct ort_abi_epctx_generation_disabled_for_auto_meta_device
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {
  std::vector<std::string> additional_required_devices() const override { return {"AUTO:" + GetParam()}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_epctx_generation_disabled_for_auto_meta_device, run) {
  const auto& physical_device = GetParam();
  const std::string ov_device_name = "AUTO:" + physical_device;

  const auto& model_info = model_library_t::GetModel("mul_1");
  fs::path epctx_model = temp_dir_ / ("mul_1_ctx_AUTO_" + physical_device + ".onnx");

  device_t device(ov_device_name);
  auto network = network_t::create(model_info.model_path_);
  network.ort_session_config_.insert(
      {{kOrtSessionOptionEpContextEnable, "1"},
       {kOrtSessionOptionEpContextFilePath, epctx_model.string()}});

  EXPECT_THROW({ graph_t graph(device, network); }, Ort::Exception);
  EXPECT_FALSE(fs::exists(epctx_model));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_epctx_generation_disabled_for_auto_meta_device,
    testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
    [](const testing::TestParamInfo<ort_abi_epctx_generation_disabled_for_auto_meta_device::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_generate_and_run_epctx_model_cross_ep
    : public ort_abi_generate_and_run_epctx_model_base,
      public testing::WithParamInterface<std::tuple<std::string, embed_mode_t, ep_type_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
  bool requires_legacy_ep() const override { return true; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_generate_and_run_epctx_model_cross_ep, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());
  const auto& generator_ep = std::get<2>(GetParam());
  const auto runner_ep = get_opposite_ep(generator_ep);

  run_test(ov_device_name, embed_mode, generator_ep, runner_ep);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_generate_and_run_epctx_model_cross_ep,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off),
        testing::Values(ep_type_t::legacy, ep_type_t::abi) /* which ep generates the ep context */
        ),
    [](const testing::TestParamInfo<ort_abi_generate_and_run_epctx_model_cross_ep::ParamType>& info) {
      auto runner_ep = ortAbiTestFixture::get_opposite_ep(std::get<2>(info.param));
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)) + "_gen_" +
             to_string(std::get<2>(info.param)) + "_run_" + to_string(runner_ep);
    });

///////////////////////////////////////////////////////////////////////
// Base test implementation that can be shared between fixtures
struct ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_base : public ortAbiTestFixture {
  void run_test(
      const std::string& ov_device_name,
      embed_mode_t embed_mode,
      session_model_load_mode_t ep_gen_mode,
      session_model_load_mode_t ep_load_mode,
      weight_sharing_t weight_sharing,
      ep_type_t generator_ep,
      ep_type_t runner_ep) {
    const auto& model_info = model_library_t::GetModel("gemm_external");

    // Want 3 different locations for the following:
    // 1. Model file
    // 2. Model external weights file (when using memory to load weights)
    // 3. ep context model
    fs::path epctx_model = "gemm_external_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx";
    epctx_model = temp_dir_ / "epctx" / epctx_model;
    fs::path model_path = temp_dir_ / model_info.model_path_.filename();
    fs::path original_model_weights = model_info.external_data_path_;
    fs::path external_weights_dir = temp_dir_ / "gemm_external_ctx";

    fs::create_directory(epctx_model.parent_path());
    fs::copy_file(model_info.model_path_, model_path);

    auto external_weights_setup = [&](session_model_load_mode_t mode, bool generating_ep_context) {
      ASSERT_NE(epctx_model.parent_path(), fs::path("testdata"));
      ASSERT_NE(model_path.parent_path(), fs::path("testdata"));
      if (mode == session_model_load_mode_t::memory) {
        // Remove any existing weights files to ensure they are loaded from the external weights path
        fs::remove(epctx_model.parent_path() / original_model_weights.filename());
        fs::remove(model_path.parent_path() / original_model_weights.filename());

        fs::create_directory(external_weights_dir);
        fs::copy_file(
            original_model_weights,
            external_weights_dir / original_model_weights.filename(),
            fs::copy_options::overwrite_existing);
      } else {
        // make sure external weights dir is clean to ensure it's not used
        fs::remove_all(external_weights_dir);

        fs::path weights_path = epctx_model.parent_path() / original_model_weights.filename();
        if (generating_ep_context) {
          weights_path = model_path.parent_path() / original_model_weights.filename();
        }

        fs::copy_file(original_model_weights, weights_path, fs::copy_options::overwrite_existing);
      }
    };

    // Generate epctx model
    {
      SetProvider(generator_ep);
      bool generating_ep_context = true;
      external_weights_setup(ep_gen_mode, generating_ep_context);

      auto network = network_t::create(model_path, ep_gen_mode);
      network.ort_session_config_.insert(
          {{kOrtSessionOptionEpContextEnable, "1"},
           {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
           {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)},
           {kOrtSessionOptionShareEpContexts, to_string(weight_sharing)}});
      if (ep_gen_mode == session_model_load_mode_t::memory) {
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath,
            external_weights_dir.string());
      }

      model_info.create_session(ov_device_name, std::move(network));

      EXPECT_TRUE(fs::exists(epctx_model));
    }

    // Run epctx model
    {
      SetProvider(runner_ep);
      bool generating_ep_context = false;
      external_weights_setup(ep_load_mode, generating_ep_context);
      auto network = network_t::create(epctx_model, ep_load_mode);
      network.ort_session_config_.emplace(kOrtSessionOptionShareEpContexts, to_string(weight_sharing));
      if (ep_load_mode == session_model_load_mode_t::memory) {
        network.ort_session_config_.emplace(
            kOrtSessionOptionsModelExternalInitializersFileFolderPath,
            external_weights_dir.string());
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, epctx_model.string());
      }

      auto session = model_info.create_session(ov_device_name, std::move(network));
      session->run_and_validate();
    }
  }
};

///////////////////////////////////////////////////////////////////////
struct ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants
    : public ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_base,
      public testing::WithParamInterface<std::tuple<
          std::string,
          embed_mode_t,
          session_model_load_mode_t,
          session_model_load_mode_t,
          weight_sharing_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());
  const auto& ep_gen_mode = std::get<2>(GetParam());
  const auto& ep_load_mode = std::get<3>(GetParam());
  const auto& weight_sharing = std::get<4>(GetParam());
  if (embed_mode == embed_mode_t::on && weight_sharing == weight_sharing_t::on) {
    GTEST_SKIP() << "Skipping unsupported combination of embed_mode=on and weight_sharing=on";
  }

  run_test(ov_device_name, embed_mode, ep_gen_mode, ep_load_mode, weight_sharing, ep_type_t::abi, ep_type_t::abi);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off),
        testing::Values(
            session_model_load_mode_t::memory,
            session_model_load_mode_t::file),                                                /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file), /* ep context loading */
        testing::Values(weight_sharing_t::on, weight_sharing_t::off)),
    [](const testing::TestParamInfo<ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants::ParamType>&
           info) {
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)) + "_gen_" +
             to_string(std::get<2>(info.param)) + "_run_" + to_string(std::get<3>(info.param)) +
             "_weight_share_" + to_string(std::get<4>(info.param));
    });

///////////////////////////////////////////////////////////////////////
struct ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_cross_ep
    : public ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_base,
      public testing::WithParamInterface<std::tuple<
          std::string,
          embed_mode_t,
          session_model_load_mode_t,
          session_model_load_mode_t,
          weight_sharing_t,
          ep_type_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
  bool requires_legacy_ep() const override { return true; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_cross_ep, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());
  const auto& ep_gen_mode = std::get<2>(GetParam());
  const auto& ep_load_mode = std::get<3>(GetParam());
  const auto& weight_sharing = std::get<4>(GetParam());
  const auto& generator_ep = std::get<5>(GetParam());
  const auto runner_ep = get_opposite_ep(generator_ep);

  if (embed_mode == embed_mode_t::on && weight_sharing == weight_sharing_t::on) {
    GTEST_SKIP() << "Skipping unsupported combination of embed_mode=on and weight_sharing=on";
  }

  run_test(ov_device_name, embed_mode, ep_gen_mode, ep_load_mode, weight_sharing, generator_ep, runner_ep);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_cross_ep,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off),
        testing::Values(
            session_model_load_mode_t::memory,
            session_model_load_mode_t::file),                                                /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file), /* ep context loading */
        testing::Values(weight_sharing_t::on, weight_sharing_t::off),
        testing::Values(ep_type_t::legacy, ep_type_t::abi) /* which ep generates the ep context */
        ),
    [](const testing::TestParamInfo<
        ort_abi_generate_and_run_epctx_model_ext_weights_pathing_variants_cross_ep::ParamType>& info) {
      auto runner_ep = ortAbiTestFixture::get_opposite_ep(std::get<5>(info.param));
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)) + "_gen_" +
             to_string(std::get<2>(info.param)) + "_run_" + to_string(std::get<3>(info.param)) +
             "_weight_share_" + to_string(std::get<4>(info.param)) + "_gen_" +
             to_string(std::get<5>(info.param)) + "_run_" + to_string(runner_ep);
    });

////////////////////////////////////////////////////////////
// Base test implementation that can be shared between fixtures
struct ort_abi_generate_and_run_epctx_model_pathing_variants_base : public ortAbiTestFixture {
  void run_test(
      const std::string& ov_device_name,
      embed_mode_t embed_mode,
      session_model_load_mode_t ep_gen_mode,
      session_model_load_mode_t ep_load_mode,
      ep_type_t generator_ep,
      ep_type_t runner_ep) {
    fs::path epctx_model = "mul_1_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx";
    epctx_model = temp_dir_ / epctx_model;
    const auto& model_info = model_library_t::GetModel("mul_1");
    fs::path model_path = model_info.model_path_;

    // Generate epctx model
    {
      SetProvider(generator_ep);
      auto network = network_t::create(model_path, ep_gen_mode);

      network.ort_session_config_.insert(
          {{kOrtSessionOptionEpContextEnable, "1"},
           {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
           {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)}});

      model_info.create_session(ov_device_name, std::move(network));

      EXPECT_TRUE(fs::exists(epctx_model));
    }

    // Run epctx model
    {
      SetProvider(runner_ep);
      auto network = network_t::create(epctx_model, ep_load_mode);

      if (ep_load_mode == session_model_load_mode_t::memory) {
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, epctx_model.string());
      }
      auto session = model_info.create_session(ov_device_name, std::move(network));
      session->run_and_validate();
    }
  }
};

////////////////////////////////////////////////////////////
struct ort_abi_generate_and_run_epctx_model_pathing_variants
    : public ort_abi_generate_and_run_epctx_model_pathing_variants_base,
      public testing::WithParamInterface<
          std::tuple<std::string, embed_mode_t, session_model_load_mode_t, session_model_load_mode_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_generate_and_run_epctx_model_pathing_variants, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());
  const auto& ep_gen_mode = std::get<2>(GetParam());
  const auto& ep_load_mode = std::get<3>(GetParam());

  run_test(ov_device_name, embed_mode, ep_gen_mode, ep_load_mode, ep_type_t::abi, ep_type_t::abi);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_generate_and_run_epctx_model_pathing_variants,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off),
        testing::Values(
            session_model_load_mode_t::memory,
            session_model_load_mode_t::file),                                               /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file) /* ep context loading */
        ),
    [](const testing::TestParamInfo<ort_abi_generate_and_run_epctx_model_pathing_variants::ParamType>& info) {
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)) + "_gen_" +
             to_string(std::get<2>(info.param)) + "_run_" + to_string(std::get<3>(info.param));
    });

///////////////////////////////////////////////////////////////////////
// Base class for ovir tests with shared run_test method
struct ort_abi_run_epctx_ovir_model_base : public ortAbiTestFixture {
  ///////////////////////////////////////////////////////////////////////////
  void run_test(
      const std::string& ov_device_name,
      session_model_load_mode_t ep_gen_mode,
      session_model_load_mode_t ep_load_mode,
      weight_sharing_t weight_sharing,
      ep_type_t generator_ep,
      ep_type_t runner_ep) {
    fs::path epctx_model = "mul_1_ctx_" + ov_device_name + ".onnx";
    epctx_model = temp_dir_ / epctx_model;
    const auto& model_info = model_library_t::GetModel("mul_1_ep_ctx_ovir");
    fs::path model_path = model_info.model_path_;

    // Generate epctx model
    {
      SetProvider(generator_ep);
      auto network = network_t::create(model_path, ep_gen_mode);

      network.ort_session_config_.insert(
          {{kOrtSessionOptionEpContextEnable, "1"},
           {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
           {kOrtSessionOptionShareEpContexts, to_string(weight_sharing)}});

      model_info.create_session(ov_device_name, std::move(network));

      EXPECT_TRUE(fs::exists(epctx_model));
    }

    // Run epctx model
    {
      SetProvider(runner_ep);
      auto network = network_t::create(epctx_model, ep_load_mode);

      network.ort_session_config_.emplace(kOrtSessionOptionShareEpContexts, to_string(weight_sharing));
      if (ep_load_mode == session_model_load_mode_t::memory) {
        network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, epctx_model.string());
      }

      auto session = model_info.create_session(ov_device_name, std::move(network));
      session->run_and_validate();
    }
  }
};

///////////////////////////////////////////////////////////////////////////
struct ort_abi_gen_and_run_epctx_ovir_model
    : public ort_abi_run_epctx_ovir_model_base,
      public testing::WithParamInterface<
          std::tuple<std::string, session_model_load_mode_t, session_model_load_mode_t, weight_sharing_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_gen_and_run_epctx_ovir_model, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& ep_gen_mode = std::get<1>(GetParam());
  const auto& ep_load_mode = std::get<2>(GetParam());
  const auto& weight_sharing = std::get<3>(GetParam());

  const auto generator_ep = ep_type_t::abi;
  const auto runner_ep = ep_type_t::abi;

  run_test(ov_device_name, ep_gen_mode, ep_load_mode, weight_sharing, generator_ep, runner_ep);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_gen_and_run_epctx_ovir_model,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(
            /*session_model_load_mode_t::memory, cannot create ep context from memory with ovir since
          kOrtSessionOptionEpContextFilePath is overloaded in ORT api. */
            session_model_load_mode_t::file),                                                /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file), /* ep context loading */
        testing::Values(weight_sharing_t::on, weight_sharing_t::off)),
    [](const testing::TestParamInfo<ort_abi_gen_and_run_epctx_ovir_model::ParamType>& info) {
      return std::get<0>(info.param) + "_" + to_string(std::get<1>(info.param)) + "_run_" +
             to_string(std::get<2>(info.param)) + "_weight_share_" + to_string(std::get<3>(info.param));
    });

///////////////////////////////////////////////////////////////////////////
// Cross-EP variant (generator / runner can differ)
struct ort_abi_gen_and_run_epctx_ovir_model_cross_ep
    : public ort_abi_run_epctx_ovir_model_base,
      public testing::WithParamInterface<
          std::tuple<std::string, session_model_load_mode_t, session_model_load_mode_t, weight_sharing_t, ep_type_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
  bool requires_legacy_ep() const override { return true; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_gen_and_run_epctx_ovir_model_cross_ep, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& ep_gen_mode = std::get<1>(GetParam());
  const auto& ep_load_mode = std::get<2>(GetParam());
  const auto& weight_sharing = std::get<3>(GetParam());
  const auto& generator_ep = std::get<4>(GetParam());
  const auto runner_ep = get_opposite_ep(generator_ep);

  run_test(ov_device_name, ep_gen_mode, ep_load_mode, weight_sharing, generator_ep, runner_ep);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_gen_and_run_epctx_ovir_model_cross_ep,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(
            /*session_model_load_mode_t::memory, cannot create ep context from memory with ovir since
          kOrtSessionOptionEpContextFilePath is overloaded in ORT api. */
            session_model_load_mode_t::file),                                                /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file), /* ep context loading */
        testing::Values(weight_sharing_t::on, weight_sharing_t::off),
        testing::Values(ep_type_t::legacy, ep_type_t::abi) /* which ep generates the ep context */
        ),
    [](const testing::TestParamInfo<ort_abi_gen_and_run_epctx_ovir_model_cross_ep::ParamType>& info) {
      auto runner_ep = ortAbiTestFixture::get_opposite_ep(std::get<4>(info.param));
      return std::get<0>(info.param) + "_" + to_string(std::get<1>(info.param)) + "_" +
             to_string(std::get<2>(info.param)) + "_weight_share_" + to_string(std::get<3>(info.param)) +
             "_gen_" + to_string(std::get<4>(info.param)) + "_run_" + to_string(runner_ep);
    });

///////////////////////////////////////////////////////////////////////////
// Simple test that just runs an existing EP context model (no generation)
struct ort_abi_run_epctx_ovir_model
    : public ort_abi_run_epctx_ovir_model_base,
      public testing::WithParamInterface<std::tuple<std::string, weight_sharing_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_run_epctx_ovir_model, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& weight_sharing = std::get<1>(GetParam());
  const auto& model_info = model_library_t::GetModel("mul_1_ep_ctx_ovir");

  auto network = network_t::create(model_info.model_path_);
  network.ort_session_config_.emplace(kOrtSessionOptionShareEpContexts, to_string(weight_sharing));
  model_info.create_session(ov_device_name, std::move(network))->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
struct ort_abi_epctx_capability : public ortAbiTestFixture {};

TEST_F(ort_abi_epctx_capability, appended_supported_node_not_claimed_by_ov) {
  const auto& model_info = model_library_t::GetModel("mul_1_ep_ctx_ovir");
  session_options_wrapper_t session_options(
      device_t("CPU"),
      {{kOrtSessionOptionsDisableCPUEPFallback, "0"}, {kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1"}});
  auto session = Ort::Session::CreateModelEditorSession(
      env_, model_info.model_path_.native().c_str(), session_options);

  auto graph_outputs = session.GetOutputs();
  ASSERT_EQ(graph_outputs.size(), 1u);
  ASSERT_EQ(graph_outputs[0].GetName(), std::string("Y"));

  Ort::Graph graph;
  Ort::Node relu("Relu", "", "appended_relu", {"Y"}, {"Y_after_relu"});
  graph.AddNode(relu);
  graph_outputs[0] = Ort::ValueInfo("Y_after_relu", graph_outputs[0].TypeInfo());
  graph.SetOutputs(graph_outputs);

  std::vector<Ort::Model::DomainOpsetPair> opsets;
  Ort::Model augmentation(opsets);
  augmentation.AddGraph(graph);
  session.FinalizeModelEditorSession(augmentation, session_options);

  const auto assignments = session.GetEpGraphAssignmentInfo();
  const Ort::ConstEpAssignedSubgraph* openvino_assignment = nullptr;
  const Ort::ConstEpAssignedSubgraph* cpu_assignment = nullptr;
  for (const auto& assignment : assignments) {
    const std::string ep_name = assignment.GetEpName();
    if (ep_name == "OpenVINOExecutionProvider") {
      ASSERT_EQ(openvino_assignment, nullptr);
      openvino_assignment = &assignment;
    } else if (ep_name == "CPUExecutionProvider") {
      ASSERT_EQ(cpu_assignment, nullptr);
      cpu_assignment = &assignment;
    }
  }

  ASSERT_NE(openvino_assignment, nullptr);
  const auto openvino_nodes = openvino_assignment->GetNodes();
  ASSERT_EQ(openvino_nodes.size(), 1u);
  EXPECT_EQ(openvino_nodes[0].GetDomain(), std::string("com.microsoft"));
  EXPECT_EQ(openvino_nodes[0].GetOperatorType(), std::string("EPContext"));

  ASSERT_NE(cpu_assignment, nullptr);
  const auto cpu_nodes = cpu_assignment->GetNodes();
  ASSERT_EQ(cpu_nodes.size(), 1u);
  EXPECT_EQ(cpu_nodes[0].GetOperatorType(), std::string("Relu"));

  std::vector<float> input(6, 2.f);
  const std::vector<int64_t> input_shape{3, 2};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y_after_relu"};
  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

  ASSERT_EQ(outputs.size(), 1u);
  const auto* output = outputs[0].GetTensorData<float>();
  EXPECT_THAT(std::vector<float>(output, output + 6), testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

TEST_F(ort_abi_epctx_capability, foreign_epctx_node_is_not_claimed) {
  const auto& model_info = model_library_t::GetModel("mul_1_ep_ctx_ovir");
  const std::vector<int64_t> shape{3, 2};
  Ort::TensorTypeAndShapeInfo tensor_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
  auto type_info = Ort::TypeInfo::CreateTensorInfo(tensor_info.GetConst());

  std::vector<Ort::ValueInfo> graph_inputs;
  graph_inputs.emplace_back("X", type_info.GetConst());
  std::vector<Ort::ValueInfo> graph_outputs;
  graph_outputs.emplace_back("Y", type_info.GetConst());

  int64_t embed_mode = 0;
  int64_t main_context = 1;
  // A valid OVIR payload makes the negative control fail only when OpenVINO incorrectly claims the foreign node.
  const std::string context_path = path_to_utf8(model_info.model_path_.parent_path() / "mul_1_ep_ctx_ovir.xml");
  const std::string source = "ForeignExecutionProvider";
  ASSERT_LE(context_path.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
  ASSERT_LE(source.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
  std::vector<Ort::OpAttr> attributes;
  attributes.emplace_back("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT);
  attributes.emplace_back("main_context", &main_context, 1, ORT_OP_ATTR_INT);
  attributes.emplace_back("ep_cache_context", context_path.c_str(), static_cast<int>(context_path.size()),
                          ORT_OP_ATTR_STRING);
  attributes.emplace_back("source", source.c_str(), static_cast<int>(source.size()), ORT_OP_ATTR_STRING);

  Ort::Graph graph;
  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);
  Ort::Node ep_context("EPContext", "com.microsoft", "foreign_epctx", {"X"}, {"Y"}, attributes);
  graph.AddNode(ep_context);

  std::vector<Ort::Model::DomainOpsetPair> opsets{{"", 11}, {"com.microsoft", 1}};
  Ort::Model model(opsets);
  model.AddGraph(graph);

  session_options_wrapper_t session_options{device_t("CPU")};
  EXPECT_THROW({ Ort::Session session(env_, model, session_options); }, Ort::Exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_run_epctx_ovir_model,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(weight_sharing_t::on, weight_sharing_t::off)),
    [](const testing::TestParamInfo<ort_abi_run_epctx_ovir_model::ParamType>& info) { return std::get<0>(info.param) + "_weight_share_" + to_string(std::get<1>(info.param)); });

///////////////////////////////////////////////////////////////////////
struct ort_abi_bypass_umd_caching
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::pair<std::string, embed_mode_t>> {
  std::vector<std::string> additional_required_devices() const override { return {GetParam().first}; }

  fs::path cache_dir_{};

  ///////////////////////////////////////////////////////////////////////////
  virtual void SetUp(void) {
    ortAbiTestFixture::SetUp();

    fs::path local_app_data = ovep_tests::get_environment_variable("LocalAppData").value_or("");
    cache_dir_ = local_app_data / "Intel" / "NPU";

    fs::remove_all(cache_dir_);
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_bypass_umd_caching, run) {
  const auto& ov_device_name = GetParam().first;

  const auto& embed_mode = GetParam().second;
  const auto epctx_model =
      path_to_utf8(temp_dir_ / ("mul_1_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx"));

  const auto& model_info = model_library_t::GetModel("mul_1");
  session_options_wrapper_t session_options(
      ov_device_name,
      {{kOrtSessionOptionEpContextEnable, "1"},
       {kOrtSessionOptionEpContextFilePath, epctx_model.c_str()},
       {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode).c_str()}});

  model_info.create_session(std::move(session_options))->run_and_validate();

  // Cache location should be empty after session is destroyed
  auto count_regular_files = [&]() -> size_t {
    size_t count = 0;
    if (fs::exists(cache_dir_)) {
      for (fs::recursive_directory_iterator itr(cache_dir_), end; itr != end; ++itr) {
        // Compiled blobs are under <CompilerVersion>\<AppName>
        // Avoid compiler properties cached in <CompilerVersion>
        if (fs::is_regular_file(itr->status()) && itr.depth() > 1) {
          OVEP_TESTS_LOG_WARN << "Found cached file: " << itr->path();

          count++;
        }
      }
    }

    return count;
  };

  EXPECT_EQ(count_regular_files(), 0);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_bypass_umd_caching,
    testing::Values(std::make_pair("NPU", embed_mode_t::on), std::make_pair("NPU", embed_mode_t::off)),
    [](const testing::TestParamInfo<ort_abi_bypass_umd_caching::ParamType>& info) { return info.param.first + "_embed_" + to_string(info.param.second); });

///////////////////////////////////////////////////////////////////////
// Load pre-compiled EPContext models from testdata that were generated with an
// OLDER OpenVINO SDK version. The EP does not version-gate EPContext nodes for
// any device, so every discovered blob passes the EP guard and reaches the
// OpenVINO plugin's import_model(), which owns blob compatibility.
//
// Per-device expectation at import_model():
//   - NPU: backward-compatible blob format (2025.4+) -> imports and runs.
//   - CPU: exact requirements-block match (magic + OV version + ISA); an
//     older-SDK blob is rejected GRACEFULLY the same way as GPU.
//   - GPU: full-descriptor equality (OV version + driver + GFX IP + EUs); an
//     older-SDK blob mismatches on the OV-version component and is rejected
//     GRACEFULLY -- session creation throws Ort::Exception, no crash.
//
// USAGE: drop pre-compiled models into testdata with naming pattern:
//   <base_model>_ctx_sdk_<version>_<device>_embed_<on|off>.onnx
// e.g. testdata/mul_1_ctx_sdk_2025_4_NPU_embed_on.onnx
// Tests auto-discover and activate without code changes.
struct ort_abi_load_cross_sdk_version_epctx_models
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, embed_mode_t, std::string>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }

  static std::vector<std::tuple<std::string, embed_mode_t, std::string>> discover_models(
      const std::string& base_model_name) {
    std::vector<std::tuple<std::string, embed_mode_t, std::string>> models;
    const std::vector<std::string> devices = {"CPU", "GPU", "NPU"};
    const std::vector<embed_mode_t> embed_modes = {embed_mode_t::on, embed_mode_t::off};
    const std::vector<std::string> sdk_versions = {"2025_2", "2025_3", "2025_4", "2026_0", "2026_1", "2026_2"};

    for (const auto& device : devices) {
      for (const auto& embed_mode : embed_modes) {
        // Use "on"/"off" instead of "1"/"0" to match actual file naming convention
        std::string embed_str = (embed_mode == embed_mode_t::on) ? "on" : "off";

        for (const auto& sdk_ver : sdk_versions) {
          fs::path model_path = fs::path("testdata") / (base_model_name + "_ctx_sdk_" + sdk_ver + "_" +
                                                        device + "_embed_" + embed_str + ".onnx");
          if (fs::exists(model_path)) {
            models.emplace_back(device, embed_mode, model_path.string());
          }
        }
      }
    }

    return models;
  }
};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_load_cross_sdk_version_epctx_models, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& epctx_model_path = std::get<2>(GetParam());

  device_t device(ov_device_name);

  network_t network{};
  data_utils::read_binary_file(epctx_model_path, network.buffer_);

  // Always specify external data path for both embed modes
  // (EP context blob location for embed_mode::off, model weights for both)
  network.ort_session_config_.insert(
      {{kOrtSessionOptionsModelExternalInitializersFileFolderPath,
        fs::path(epctx_model_path).parent_path().string()}});

  // GPU and CPU have no cross-version blob compatibility. The older-SDK blob
  // still passes the EP guard (the EP no longer version-gates any device) but
  // the OpenVINO plugin rejects it at import_model() via exact-match
  // requirements checks. The rejection must be graceful: session creation
  // throws Ort::Exception rather than crashing.
  if (ov_device_name == "GPU" || ov_device_name == "CPU") {
    EXPECT_THROW({ graph_t graph(device, network); }, Ort::Exception);
    return;
  }

  // NPU: the blob format is backward-compatible across SDK versions -> load and run.
  graph_t graph(device, network);
  pipeline_t pipeline(device, graph);

  std::map<std::string, std::vector<float>> input_data;
  input_data.try_emplace("X", std::vector<float>(6, 2.f));

  std::map<std::string, std::vector<float>> output_data;
  output_data.try_emplace("Y", std::vector<float>(6));

  pipeline.run(input_data, output_data);

  // Correct output proves the older-SDK EPContext blob loaded and ran.
  EXPECT_THAT(output_data.at("Y"), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    ort_abi_load_cross_sdk_version_epctx_models,
    testing::ValuesIn(ort_abi_load_cross_sdk_version_epctx_models::discover_models("mul_1")),
    [](const testing::TestParamInfo<ort_abi_load_cross_sdk_version_epctx_models::ParamType>& info) {
      auto model_name = fs::path(std::get<2>(info.param)).stem().string();
      return std::get<0>(info.param) + "_" + model_name;
    });

// TODO: ericcraw reduce copy paste
struct gen_and_run_paritioned_ep_context_ext_weights_pathing_variants
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<
          std::string,
          embed_mode_t,
          session_model_load_mode_t,
          session_model_load_mode_t,
          weight_sharing_t>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

TEST_P(gen_and_run_paritioned_ep_context_ext_weights_pathing_variants, run) {
  const auto& ov_device_name = std::get<0>(GetParam());
  const auto& embed_mode = std::get<1>(GetParam());
  const auto& ep_gen_mode = std::get<2>(GetParam());
  const auto& ep_load_mode = std::get<3>(GetParam());
  const auto& weight_sharing = std::get<4>(GetParam());

  if (embed_mode == embed_mode_t::on && weight_sharing == weight_sharing_t::on) {
    GTEST_SKIP() << "Skipping unsupported combination of embed_mode=on and weight_sharing=on";
  }

  // Setup paths
  const auto& model_info = model_library_t::GetModel("gemm_custom_op_external");
  fs::path epctx_model = "gemm_custom_op_ctx_" + ov_device_name + "_embed_" + to_string(embed_mode) + ".onnx";
  epctx_model = temp_dir_ / "epctx" / epctx_model;
  fs::path model_path = temp_dir_ / "gemm_custom_op_external.onnx";
  fs::path original_model_weights = model_info.external_data_path_;
  fs::path external_weights_dir = temp_dir_ / "gemm_custom_op_external_ctx";

  fs::create_directory(epctx_model.parent_path());
  fs::copy_file(model_info.model_path_, model_path, fs::copy_options::overwrite_existing);

  auto external_weights_setup = [&](session_model_load_mode_t mode, bool generating_ep_context) {
    ASSERT_NE(epctx_model.parent_path(), fs::path("testdata"));
    ASSERT_NE(model_path.parent_path(), fs::path("testdata"));
    if (mode == session_model_load_mode_t::memory) {
      // Remove any existing weights files to ensure they are loaded from the external weights path
      fs::remove(epctx_model.parent_path() / original_model_weights.filename());
      fs::remove(model_path.parent_path() / original_model_weights.filename());

      fs::create_directory(external_weights_dir);
      fs::copy_file(
          original_model_weights,
          external_weights_dir / original_model_weights.filename(),
          fs::copy_options::overwrite_existing);
    } else {
      // make sure external weights dir is clean to ensure it's not used
      fs::remove_all(external_weights_dir);

      fs::path weights_path = epctx_model.parent_path() / original_model_weights.filename();
      if (generating_ep_context) {
        weights_path = model_path.parent_path() / original_model_weights.filename();
      }

      fs::copy_file(original_model_weights, weights_path, fs::copy_options::overwrite_existing);
    }
  };

  // Generate epctx model
  {
    bool generating_ep_context = true;
    external_weights_setup(ep_gen_mode, generating_ep_context);
    auto network = network_t::create(model_path, ep_gen_mode);
    network.set_cpu_fallback(true);

    network.ort_session_config_.insert(
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, epctx_model.string()},
         {kOrtSessionOptionEpContextEmbedMode, to_string(embed_mode)},
         {kOrtSessionOptionShareEpContexts, to_string(weight_sharing)}});
    if (ep_gen_mode == session_model_load_mode_t::memory) {
      network.ort_session_config_.emplace(
          kOrtSessionOptionsModelExternalInitializersFileFolderPath,
          external_weights_dir.string());
    }

    device_t device(ov_device_name);
    graph_t graph(device, network);

    EXPECT_TRUE(fs::exists(epctx_model));
  }

  // Run epctx model
  {
    bool generating_ep_context = false;
    external_weights_setup(ep_load_mode, generating_ep_context);
    auto network = network_t::create(epctx_model, ep_load_mode);
    network.set_cpu_fallback(true);

    network.ort_session_config_.emplace(kOrtSessionOptionShareEpContexts, to_string(weight_sharing));
    if (ep_load_mode == session_model_load_mode_t::memory) {
      network.ort_session_config_.emplace(
          kOrtSessionOptionsModelExternalInitializersFileFolderPath,
          external_weights_dir.string());
      network.ort_session_config_.emplace(kOrtSessionOptionEpContextFilePath, epctx_model.string());
    }

    device_t device(ov_device_name);
    graph_t graph(device, network);
    pipeline_t pipeline(device, graph);

    auto input_data = model_info.load_input_data<std::byte>();
    auto expected_output = model_info.load_expected_output_data<std::byte>();

    // Create output buffers with correct sizes
    io_byte_data_t output_data;
    for (const auto& [name, expected] : expected_output) {
      output_data[name].resize(expected.size());
    }

    pipeline.run(input_data, output_data);

    // Validate using model info's validation logic
    model_info.validate_outputs(output_data, expected_output);
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context,
    gen_and_run_paritioned_ep_context_ext_weights_pathing_variants,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(embed_mode_t::on, embed_mode_t::off),
        testing::Values(
            session_model_load_mode_t::memory,
            session_model_load_mode_t::file),                                                /* ep context generation */
        testing::Values(session_model_load_mode_t::memory, session_model_load_mode_t::file), /* ep context loading */
        testing::Values(weight_sharing_t::on, weight_sharing_t::off)),
    [](const testing::TestParamInfo<gen_and_run_paritioned_ep_context_ext_weights_pathing_variants::ParamType>&
           info) {
      return std::get<0>(info.param) + "_embed_" + to_string(std::get<1>(info.param)) + "_gen_" +
             to_string(std::get<2>(info.param)) + "_run_" + to_string(std::get<3>(info.param)) +
             "_weight_share_" + to_string(std::get<4>(info.param));
    });
