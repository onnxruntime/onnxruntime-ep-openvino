// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"
#include "ov_test_version.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "ort_builtin_model_library.h"
#include "path_utils.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;
using model_library_t = ortAbiTestFixture::model_library_t;

///////////////////////////////////////////////////////////////////////
// tiny_llama3 model constants (from config.json):
//   vocab=128256
static constexpr int64_t kVocabSize = 128256;

///////////////////////////////////////////////////////////////////////
// Create session options with enable_causallm=1 for the given device.
static session_options_t make_stateful_opts(const std::string& device) {
  return session_options_t(device_t(device), {}, {{"enable_causallm", "1"}});
}

///////////////////////////////////////////////////////////////////////
// Run one inference step on the tiny_llama3 stateful model.
//
// Builds inputs:
//   input_ids      int64 [1, seq]
//   attention_mask int64 [1, past_seq+seq]  — all ones
//   position_ids   int64 [1, seq]           — range [past_seq, past_seq+seq)
//
// past_key_values are intentionally omitted: after MakeStateful the EP removes
// them from both the input and output lists; the KV cache is managed as internal
// OV model state.
//
// Returns a flat vector of kVocabSize floats — OV stateful models always emit
// [1, 1, vocab] (last token only), regardless of the prefill length.
static std::vector<float> run_step(
    graph_t& graph,
    const std::vector<int64_t>& ids,
    int64_t past_seq = 0,
    bool provide_output = true) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const int64_t seq = static_cast<int64_t>(ids.size());
  const int64_t total_seq = past_seq + seq;

  // input_ids: int64 [1, seq]
  const std::array<int64_t, 2> ids_shape{1, seq};
  auto ids_val = Ort::Value::CreateTensor<int64_t>(
      mem,
      const_cast<int64_t*>(ids.data()), static_cast<size_t>(seq),
      ids_shape.data(), ids_shape.size());

  // attention_mask: int64 [1, past_seq+seq], all ones
  const std::array<int64_t, 2> mask_shape{1, total_seq};
  std::vector<int64_t> mask_data(static_cast<size_t>(total_seq), 1LL);
  auto mask_val = Ort::Value::CreateTensor<int64_t>(
      mem,
      mask_data.data(), mask_data.size(),
      mask_shape.data(), mask_shape.size());

  // position_ids: int64 [1, seq], range [past_seq, past_seq+seq)
  std::vector<int64_t> pos_data(static_cast<size_t>(seq));
  std::iota(pos_data.begin(), pos_data.end(), past_seq);
  auto pos_val = Ort::Value::CreateTensor<int64_t>(
      mem,
      pos_data.data(), pos_data.size(),
      ids_shape.data(), ids_shape.size());

  // NaNs expose outputs that were accepted but never written by the EP.
  // OV stateful models always emit [1, 1, vocab] regardless of prefill length.
  const size_t logits_elems = static_cast<size_t>(kVocabSize);
  std::vector<float> logits_buf(logits_elems, std::numeric_limits<float>::quiet_NaN());
  const std::array<int64_t, 3> logits_shape{1, 1, kVocabSize};
  auto logits_val = Ort::Value::CreateTensor<float>(
      mem,
      logits_buf.data(), logits_buf.size(),
      logits_shape.data(), logits_shape.size());

  // Assemble input names and values
  std::vector<const char*> in_names = {"input_ids", "attention_mask", "position_ids"};
  for (auto& name : in_names) {
    if (!graph.find_input(name)) {
      for (const auto& graph_name : graph.input_names()) {
        if (std::string(graph_name).find(name) != std::string::npos) {
          name = graph_name;
        }
      }
    }
  }

  std::vector<Ort::Value> in_vals;
  in_vals.push_back(std::move(ids_val));
  in_vals.push_back(std::move(mask_val));
  in_vals.push_back(std::move(pos_val));

  std::vector<const char*> out_names = {"logits"};
  for (auto& name : out_names) {
    if (!graph.find_output(name)) {
      for (const auto& graph_name : graph.output_names()) {
        if (std::string(graph_name).find(name) != std::string::npos) {
          name = graph_name;
        }
      }
    }
  }
  std::vector<Ort::Value> out_vals;
  if (provide_output) {
    out_vals.push_back(std::move(logits_val));
  } else {
    out_vals.emplace_back(nullptr);
  }

  graph.run(in_names, in_vals, out_names, out_vals);

  if (!provide_output) {
    const auto& output = out_vals.at(0);
    const auto output_size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    logits_buf.resize(output_size);
    std::copy_n(output.GetTensorData<float>(), output_size, logits_buf.data());
  }

  return logits_buf;
}

///////////////////////////////////////////////////////////////////////
// Inference tests — shared between the ONNX source model and the pre-compiled
// EPContext IR model (embed_mode=0, external xml+bin).
// Parameterised over {device, model_name}.
struct ort_abi_stateful_inference
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::string>> {
  void SetUp() override {
    ortAbiTestFixture::SetUp();
    if (std::get<0>(GetParam()) == "NPU" && OvTestVersion::built() >= OvTestVersion{2026, 5}) {
      GTEST_SKIP() << "NPU stateful tests are disabled with OpenVINO 2026.5";
    }
  }

  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////
// Test 1: StatefulCompile + StatefulPostCompileInit succeed without error.
TEST_P(ort_abi_stateful_inference, session_creation) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));
  ASSERT_NE(graph.session_, nullptr);
}

///////////////////////////////////////////////////////////////////////
// Test 2: Prefill pass — seq > 1.  Verifies the multi-token Compute path.
TEST_P(ort_abi_stateful_inference, prefill_inference) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  const auto logits = run_step(graph, {1, 2, 3}, /*past_seq=*/0);
  // OV stateful models always emit [1, 1, vocab] — last token only.
  ASSERT_EQ(logits.size(), static_cast<size_t>(kVocabSize));
}

///////////////////////////////////////////////////////////////////////
// Test 3: Decode step — seq == 1.  Verifies the single-token Compute path.
TEST_P(ort_abi_stateful_inference, decode_step) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  const auto logits = run_step(graph, {5}, /*past_seq=*/0);
  ASSERT_EQ(logits.size(), static_cast<size_t>(kVocabSize));
}

///////////////////////////////////////////////////////////////////////
// Test 4: Multi-step inference — prefill followed by decode steps.
// Exercises stateful KV-cache persistence across calls (OV state accumulates).
TEST_P(ort_abi_stateful_inference, multi_step_inference) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  // Prefill
  auto prefill_logits = run_step(graph, {1, 2, 3}, /*past_seq=*/0);
  ASSERT_EQ(prefill_logits.size(), static_cast<size_t>(kVocabSize));

  // Successive decode steps — past_seq grows by 1 each step to reflect
  // the accumulated context (3-token prefill + previously generated tokens).
  int64_t past_seq = 3;
  for (int64_t tok = 4; tok <= 6; ++tok, ++past_seq) {
    const auto decode_logits = run_step(graph, {tok}, past_seq);
    ASSERT_EQ(decode_logits.size(), static_cast<size_t>(kVocabSize));
  }
}

///////////////////////////////////////////////////////////////////////
// Test 5: Multiple generation sequences.
// Exercises continuous decoding (i.e. multiple prompts without resetting OV state).
TEST_P(ort_abi_stateful_inference, multi_gen_sequence) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  for (int gen = 0; gen < 2; ++gen) {
    // Prefill
    auto prefill_logits = run_step(graph, {1, 2, 3}, /*past_seq=*/0);
    ASSERT_EQ(prefill_logits.size(), static_cast<size_t>(kVocabSize));

    // Successive decode steps — past_seq grows by 1 each step to reflect
    // the accumulated context (3-token prefill + previously generated tokens).
    int64_t past_seq = 3;
    for (int64_t tok = 4; tok <= 6; ++tok, ++past_seq) {
      const auto decode_logits = run_step(graph, {tok}, past_seq);
      ASSERT_EQ(decode_logits.size(), static_cast<size_t>(kVocabSize));
    }
  }
}

TEST_P(ort_abi_stateful_inference, user_bound_output_mode_is_sticky) {
  const auto& [device, model_name] = GetParam();
  if (device == "NPU") {
    GTEST_SKIP() << "NPUW requires copying outputs from its internal requests";
  }

  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  // Dynamic output is allowed before the first external binding.
  ASSERT_EQ(run_step(graph, {1, 2}, /*past_seq=*/0, /*provide_output=*/false).size(),
            static_cast<size_t>(kVocabSize));

  // The first concrete output switches the stateful request into user-bound mode.
  ASSERT_EQ(run_step(graph, {3}, /*past_seq=*/2, /*provide_output=*/true).size(),
            static_cast<size_t>(kVocabSize));

  try {
    run_step(graph, {4}, /*past_seq=*/3, /*provide_output=*/false);
    FAIL() << "Stateful execution must reject switching from user-bound to dynamic outputs";
  } catch (const Ort::Exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("cannot switch from user-provided output buffers"), std::string::npos)
        << ex.what();
  }

  // A failed dynamic call must not reset the mode or replace the stateful request.
  ASSERT_EQ(run_step(graph, {5}, /*past_seq=*/3, /*provide_output=*/true).size(),
            static_cast<size_t>(kVocabSize));
}

TEST_P(ort_abi_stateful_inference, preallocated_logits_match_dynamic_outputs) {
  const auto& [device, model_name] = GetParam();
  const auto& mi = model_library_t::GetModel(model_name.c_str());
  auto net = network_t::create(mi.model_path_);
  auto make_options = [&]() {
    return session_options_t(device_t(device), {}, {{"enable_causallm", "1"}},
                             device == "NPU" ? std::map<std::string, std::string>{{"NPUW_SLICE_OUT", "YES"}}
                                             : std::map<std::string, std::string>{});
  };

  std::vector<std::vector<float>> expected;
  {
    graph_t graph(net, make_options());
    expected.push_back(run_step(graph, {1, 2, 3}, 0, false));
    for (int64_t tok = 4; tok <= 6; ++tok) {
      expected.push_back(run_step(graph, {tok}, tok - 1, false));
    }
  }

  graph_t graph(net, make_options());
  const auto check_logits = [&](const std::vector<float>& actual, size_t step) {
    ASSERT_EQ(actual.size(), expected[step].size());
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i])) << "step=" << step << ", logit=" << i;
      ASSERT_NEAR(actual[i], expected[step][i], 1e-4f) << "step=" << step << ", logit=" << i;
    }
  };
  check_logits(run_step(graph, {1, 2, 3}, 0, true), 0);
  for (int64_t tok = 4; tok <= 6; ++tok) {
    // NPU must also allow a dynamic output after a caller-provided one without losing KV state.
    const bool provide_output = device != "NPU" || tok != 4;
    check_logits(run_step(graph, {tok}, tok - 1, provide_output), static_cast<size_t>(tok - 3));
  }
}

///////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_stateful_compute,
    ort_abi_stateful_inference,
    testing::Combine(
        testing::Values(std::string("CPU"), std::string("GPU"), std::string("NPU")),
        testing::Values(std::string("tiny_llama3"), std::string("tiny_llama3_epctx_ir"), std::string("tiny_llama3_numeric_prefix"))),
    [](const testing::TestParamInfo<ort_abi_stateful_inference::ParamType>& info) {
      return std::get<0>(info.param) + "_" + std::get<1>(info.param);
    });

///////////////////////////////////////////////////////////////////////
// Model-specific tests — only applicable to the ONNX source model.
struct ort_abi_stateful_compute : public ortAbiSingleDeviceTestFixture {
  void SetUp() override {
    ortAbiSingleDeviceTestFixture::SetUp();
    if (GetParam() == "NPU" && OvTestVersion::built() >= OvTestVersion{2026, 5}) {
      GTEST_SKIP() << "NPU stateful tests are disabled with OpenVINO 2026.5";
    }
  }
};

///////////////////////////////////////////////////////////////////////
// Test 6: kvcache_rewind resets OV internal state; a subsequent decode step
// must succeed with the expected output size.
TEST_P(ort_abi_stateful_compute, kvcache_rewind) {
  const auto& device = GetParam();
  const auto& mi = model_library_t::GetModel("tiny_llama3");
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  // Accumulate some state with a prefill
  auto prefill_logits = run_step(graph, {1, 2, 3}, /*past_seq=*/0);
  ASSERT_EQ(prefill_logits.size(), static_cast<size_t>(kVocabSize));

  // Rewind state to index 0 (reset)
  graph.session_->set_ep_dynamic_options({{"kvcache_rewind", "0"}});

  // Decode step after rewind must succeed
  const auto decode_logits = run_step(graph, {5}, /*past_seq=*/0);
  ASSERT_EQ(decode_logits.size(), static_cast<size_t>(kVocabSize));
}

///////////////////////////////////////////////////////////////////////
// Test 7: kvcache_reorder requires src_idx/dst_idx tensors which are only
// added when should_add_kvcache_reorder=true (GPU+GQA path).  On CPU and NPU
// the reorder path is not triggered for this model, so the call must throw.
TEST_P(ort_abi_stateful_compute, kvcache_reorder_error) {
  const auto& device = GetParam();
  if (device == "GPU") {
    GTEST_SKIP() << "kvcache_reorder may be enabled on GPU for GQA models — skipping";
  }
  const auto& mi = model_library_t::GetModel("tiny_llama3");
  auto net = network_t::create(mi.model_path_);
  graph_t graph(net, make_stateful_opts(device));

  // Must throw because kvcache_reorder_enabled_ == false for CPU/NPU
  EXPECT_THROW(
      graph.session_->set_ep_dynamic_options({{"kvcache_reorder", "0;1"}}),
      Ort::Exception);
}

///////////////////////////////////////////////////////////////////////
// Test 8: EPContext round-trip — generate EPContext blob, reload, then run
// inference to verify the stateful compilation survives serialisation.
//
// Skipped on NPU: NPUW generates weightless blobs (weights are not embedded in
// the compiled blob).  import_model() on NPU requires MODEL_PTR or WEIGHTS_PATH
// alongside the blob, but ImportNativeEpContext has no access to the original
// model at reload time.  Supporting this requires storing the OV IR in the
// EPContext during generation and passing it as MODEL_PTR on reload.
TEST_P(ort_abi_stateful_compute, epctx_roundtrip) {
  const auto& device = GetParam();
  const auto& mi = model_library_t::GetModel("tiny_llama3");
  const fs::path epctx_path = temp_dir_ / "tiny_llama3_ctx.onnx";

  // --- generate EPContext ---
  {
    session_options_t gen_opts(
        device_t(device),
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_path)},
         {kOrtSessionOptionEpContextEmbedMode, "1"}},
        {{"enable_causallm", "1"}});
    auto net = network_t::create(mi.model_path_);
    graph_t gen_graph(net, std::move(gen_opts));
    ASSERT_TRUE(fs::exists(epctx_path)) << "EPContext file was not created";
  }

  // --- reload from EPContext and run inference ---
  {
    auto net2 = network_t::create(epctx_path);
    graph_t graph(net2, make_stateful_opts(device));

    const auto logits = run_step(graph, {1, 2, 3}, /*past_seq=*/0);
    ASSERT_EQ(logits.size(), static_cast<size_t>(kVocabSize));
  }
}

///////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_stateful_compute,
    ort_abi_stateful_compute,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_stateful_compute::ParamType>& info) {
      return info.param;
    });
