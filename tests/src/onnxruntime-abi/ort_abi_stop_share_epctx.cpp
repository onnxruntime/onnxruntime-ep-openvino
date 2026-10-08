// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;
namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////

using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;

struct ort_abi_stop_share_epctx_fixture : public ortAbiSingleDeviceTestFixture {};
struct ort_abi_stop_share_epctx : public ort_abi_stop_share_epctx_fixture {};

///////////////////////////////////////////////////////////////////////

// This test validates that StopShareEpContexts works as intended.
// Model Set 1: Create EP contexts from gemm_external model in temp/1 (models renamed to model1.onnx and model2.onnx)
// with shared contexts, terminated by StopShare Model Set 2: Create EP contexts from gemm_external_alternative model
// in temp/2 (models renamed to model1.onnx and model2.onnx) with shared contexts, terminated by StopShare Inference:
// Run one model from each set in an interleaved way to validate that each set uses its own weights and contexts are
// properly isolated

TEST_P(ort_abi_stop_share_epctx, run_stop_share_functionally_full) {
  const auto& device_name = GetParam();
  device_t device(device_name);

  // Create directory structure for two model sets
  const auto set1_dir = temp_dir_ / "1";
  const auto set2_dir = temp_dir_ / "2";
  fs::create_directories(set1_dir);
  fs::create_directories(set2_dir);

  auto copy_model_set = [](const std::string& base_name, const fs::path& target_dir) {
    const fs::path onnx_src = fs::path("testdata") / (base_name + ".onnx");
    const fs::path bin_src = fs::path("testdata") / (base_name + ".bin");
    fs::copy_file(onnx_src, target_dir / "model1.onnx", fs::copy_options::overwrite_existing);
    fs::copy_file(bin_src, target_dir / (base_name + ".bin"), fs::copy_options::overwrite_existing);
    fs::copy_file(onnx_src, target_dir / "model2.onnx", fs::copy_options::overwrite_existing);
  };

  // Define model set paths
  const fs::path set1_model1 = set1_dir / "model1.onnx";
  const fs::path set1_model2 = set1_dir / "model2.onnx";
  const fs::path set2_model1 = set2_dir / "model1.onnx";
  const fs::path set2_model2 = set2_dir / "model2.onnx";

  auto to_ep_context = [](const fs::path& model_path) {
    return (model_path.parent_path() / (model_path.stem().string() + "_ctx.onnx")).string();
  };

  copy_model_set("gemm_external", set1_dir);
  copy_model_set("gemm_external_alternative", set2_dir);

  // Load expected inputs and outputs for validation
  auto input_gemm = data_utils::read_binary_file<std::byte>("testdata/gemm_external.in");
  auto ref_out_gemm = data_utils::read_binary_file<std::byte>("testdata/gemm_external.out");
  auto input_alt = data_utils::read_binary_file<std::byte>("testdata/gemm_external_alternative.in");
  auto ref_out_alt = data_utils::read_binary_file<std::byte>("testdata/gemm_external_alternative.out");

  constexpr double cosine_sim_threshold = 0.97;
  ASSERT_LT(data_utils::cosine_similarity(ref_out_gemm, ref_out_alt), cosine_sim_threshold - 0.1)
      << "Reference outputs should be different for different models.";

  // struct to keep session resources alive.
  // It's important to keep them alive to ensure we can load multiple shared ep context sessions.
  struct session_container {
    std::unique_ptr<network_t> network;
    std::unique_ptr<graph_t> graph;
    std::unique_ptr<pipeline_t> pipeline;
  };

  auto run_model_and_get_cosine_similarity = [&](
                                                 const fs::path& model_path,
                                                 const std::vector<std::byte>& input_data_bytes,
                                                 const std::vector<std::byte>& expected_output,
                                                 const fs::path& epctx_model = "",
                                                 bool stop_share = false,
                                                 bool /*gen_ep_ctx*/ = false) {
    session_container session;
    session.network = std::make_unique<network_t>(network_t::create(model_path.string()));

    session.network->ort_session_config_.insert({{kOrtSessionOptionShareEpContexts, "1"}});

    if (!epctx_model.empty()) {
      session.network->ort_session_config_.insert(
          {{kOrtSessionOptionEpContextFilePath, epctx_model.string()},
           {kOrtSessionOptionEpContextEnable, "1"}});
    }

    if (stop_share) {
      session.network->ort_session_config_.insert({{kOrtSessionOptionStopShareEpContexts, "1"}});
    }

    session.graph = std::make_unique<graph_t>(device, *session.network);
    session.pipeline = std::make_unique<pipeline_t>(device, *session.graph);

    std::map<std::string, std::vector<std::byte> > input_data;
    input_data.try_emplace("input", input_data_bytes);

    std::map<std::string, std::vector<std::byte> > output_data;
    output_data.try_emplace("output", std::vector<std::byte>(expected_output.size()));

    session.pipeline->run(input_data, output_data);
    auto similarity = data_utils::cosine_similarity(output_data.at("output"), expected_output);

    return std::tuple{similarity, std::move(session)};
  };

  auto generate_model_set = [&](
                                const fs::path& model1,
                                const fs::path& model2,
                                const std::vector<std::byte>& input,
                                const std::vector<std::byte>& expected_output,
                                const std::string& set_name) {
    const bool stop = true, share = true;
    auto [sim1, _] = run_model_and_get_cosine_similarity(
        model1,
        input,
        expected_output,
        to_ep_context(model1),
        !stop,
        share);
    EXPECT_GT(sim1, cosine_sim_threshold) << set_name << " Model 1 should match expected output.";

    auto [sim2, __] =
        run_model_and_get_cosine_similarity(model2, input, expected_output, to_ep_context(model2), stop, share);
    EXPECT_GT(sim2, cosine_sim_threshold) << set_name << " Model 2 should match expected output.";
  };

  // PHASE 1 & 2: Generate both model sets with shared contexts
  generate_model_set(set1_model1, set1_model2, input_gemm, ref_out_gemm, "Set 1");
  generate_model_set(set2_model1, set2_model2, input_alt, ref_out_alt, "Set 2");

  // PHASE 3: Interleaved inference validation
  // Keep sessions alive to ensure contexts can be loaded interleaved.
  std::vector<std::tuple<double, session_container> > sessions;

  auto run_inference = [&](
                           const fs::path& model,
                           const std::vector<std::byte>& input,
                           const std::vector<std::byte>& expected_output,
                           const std::string& msg) {
    auto result = run_model_and_get_cosine_similarity(to_ep_context(model), input, expected_output);
    EXPECT_GT(std::get<0>(result), cosine_sim_threshold) << msg;
    sessions.push_back(std::move(result));
  };

  run_inference(
      set1_model1,
      input_gemm,
      ref_out_gemm,
      "Inference from Set 1 Model 1 should match gemm_external output.");
  run_inference(
      set2_model1,
      input_alt,
      ref_out_alt,
      "Inference from Set 2 Model 1 should match gemm_external_alternative output.");
  run_inference(
      set1_model2,
      input_gemm,
      ref_out_gemm,
      "Inference from Set 1 Model 2 should match gemm_external output.");
  run_inference(
      set2_model2,
      input_alt,
      ref_out_alt,
      "Inference from Set 2 Model 2 should match gemm_external_alternative output.");
}

///////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(
    ort_abi_stop_share_epctx,
    ort_abi_stop_share_epctx,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_stop_share_epctx::ParamType>& info) { return info.param; });
