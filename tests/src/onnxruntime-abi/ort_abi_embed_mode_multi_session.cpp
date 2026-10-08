// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

// Regression: embed-mode access violation when multiple sessions share a SharedContext.
// A later Deserialize() overwrites BinManager::embedded_string_ while earlier sessions'
// compiled models still reference it. Fixed by giving each EP instance its own SharedContext.
// See ov_provider.cc InitializeSharedContext.

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;

// Build a simple Y = X * W model. Caller controls dimensions/weights to vary blob size.
static Ort::Model CreateMulModel(std::vector<float>& weight_storage,
                                 const std::vector<int64_t>& dims,
                                 const std::vector<float>& weights) {
  Ort::Graph graph;

  Ort::TensorTypeAndShapeInfo tensor_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims);
  auto type_info = Ort::TypeInfo::CreateTensorInfo(tensor_info.GetConst());

  std::vector<Ort::ValueInfo> inputs;
  inputs.emplace_back("X", type_info.GetConst());

  std::vector<Ort::ValueInfo> outputs;
  outputs.emplace_back("Y", type_info.GetConst());

  graph.SetInputs(inputs);
  graph.SetOutputs(outputs);

  Ort::Node node("Mul", "", "mul_0", {"X", "W"}, {"Y"});
  graph.AddNode(node);

  weight_storage = weights;
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto w_tensor = Ort::Value::CreateTensor<float>(
      mem_info, weight_storage.data(), weight_storage.size(), dims.data(), dims.size());
  graph.AddInitializer("W", w_tensor, /*data_is_external=*/false);

  std::vector<Ort::Model::DomainOpsetPair> opsets{{"", 18}};
  Ort::Model model(opsets);
  model.AddGraph(graph);
  return model;
}

struct ort_abi_embed_mode_multi_session : public ortAbiTestFixture {
  // Uses Session(env, model, options) (ORT 1.22+) to avoid ModelCompilationOptions (ORT 1.24+).
  // Pass keep_alive to prevent the session from being destroyed immediately; keeping it alive
  // holds the SharedContextManager singleton alive across subsequent compilations.
  std::vector<uint8_t> CompileToEmbedBuffer(
      const Ort::Model& model,
      const std::string& device_name,
      std::unique_ptr<Ort::Session>* keep_alive = nullptr) {
    static int counter = 0;
    const fs::path ctx_path = temp_dir_ / ("ctx_" + std::to_string(counter++) + ".onnx");

    session_options_t so(device_name, {
                                          {kOrtSessionOptionEpContextEnable, "1"},
                                          {kOrtSessionOptionEpContextEmbedMode, "1"},
                                          {kOrtSessionOptionEpContextFilePath, ctx_path.string()},
                                      });
    Ort::Session session{env_, model, so};
    if (keep_alive) {
      *keep_alive = std::make_unique<Ort::Session>(std::move(session));
    }

    return data_utils::read_binary_file<uint8_t>(ctx_path);
  }

  // Run one inference pass with all-2.0 inputs and return the output floats.
  static std::vector<float> RunInference(Ort::Session& session, const std::vector<int64_t>& shape) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    size_t numel = 1;
    for (auto d : shape) numel *= static_cast<size_t>(d);
    std::vector<float> input_data(numel, 2.0f);

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), shape.data(), shape.size()));

    std::array<const char*, 1> input_names{"X"};
    std::array<const char*, 1> output_names{"Y"};
    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());

    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + numel);
  }
};

TEST_F(ort_abi_embed_mode_multi_session, two_sessions_access_violation_CPU) {
  const std::string device_name = "CPU";

  // Model A is large ([256,256]), Model B is small ([2,1]).
  // The size difference is key: the bug causes Session B's shorter blob to
  // replace A's larger embedded_string_, so Session A reads out-of-bounds.
  const std::vector<int64_t> dims_a{256, 256};
  const size_t numel_a = static_cast<size_t>(dims_a[0]) * static_cast<size_t>(dims_a[1]);
  std::vector<float> weights_a;
  auto model_a = CreateMulModel(weights_a, dims_a, std::vector<float>(numel_a, 1.f));

  const std::vector<int64_t> dims_b{2, 1};
  const size_t numel_b = static_cast<size_t>(dims_b[0]) * static_cast<size_t>(dims_b[1]);
  std::vector<float> weights_b;
  auto model_b = CreateMulModel(weights_b, dims_b, std::vector<float>(numel_b, 5.f));

  // Compile both models to embed-mode EP context buffers.
  auto buf_a = CompileToEmbedBuffer(model_a, device_name);
  auto buf_b = CompileToEmbedBuffer(model_b, device_name);
  ASSERT_FALSE(buf_a.empty());
  ASSERT_FALSE(buf_b.empty());
  ASSERT_GT(buf_a.size(), buf_b.size()) << "Model A's blob must be larger than Model B's for the access violation test.";

  const std::vector<float> expected_a(numel_a, 2.f);   // 2.0 * 1.0 = 2.0
  const std::vector<float> expected_b(numel_b, 10.f);  // 2.0 * 5.0 = 10.0

  // Session A works before B exists.
  session_options_t options_a(device_name);
  Ort::Session session_a(env_, buf_a.data(), buf_a.size(), options_a);

  auto result_a_before = RunInference(session_a, dims_a);
  ASSERT_EQ(result_a_before, expected_a)
      << "Session A must produce correct results before session B is created.";

  // Creating session B would overwrite the shared embedded_string_ before the fix.
  session_options_t options_b(device_name);
  Ort::Session session_b(env_, buf_b.data(), buf_b.size(), options_b);

  // Re-run session A — access violation or wrong results if the bug is present.
  auto result_a_after = RunInference(session_a, dims_a);

  EXPECT_EQ(result_a_after, expected_a)
      << "Session A results changed after Session B was created (access violation regression).";

  auto result_b = RunInference(session_b, dims_b);
  EXPECT_EQ(result_b, expected_b) << "Session B should produce correct results.";
}

// Regression: embed-mode EP context generation sessions must each create a fresh
// SharedContext to hold the embedded blob, instead of sharing one from the manager.
TEST_F(ort_abi_embed_mode_multi_session, embed_gen_context_corruption_CPU) {
  const std::string device_name = "CPU";

  const std::vector<int64_t> dims_a{256, 256};
  const size_t numel_a = static_cast<size_t>(dims_a[0]) * static_cast<size_t>(dims_a[1]);
  std::vector<float> weights_a;
  auto model_a = CreateMulModel(weights_a, dims_a, std::vector<float>(numel_a, 1.f));

  const std::vector<int64_t> dims_b{2, 1};
  const size_t numel_b = static_cast<size_t>(dims_b[0]) * static_cast<size_t>(dims_b[1]);
  std::vector<float> weights_b;
  auto model_b = CreateMulModel(weights_b, dims_b, std::vector<float>(numel_b, 5.f));

  const std::vector<float> expected_a(numel_a, 2.f);   // 2.0 * 1.0 = 2.0
  const std::vector<float> expected_b(numel_b, 10.f);  // 2.0 * 5.0 = 10.0

  // Keeping model A's compilation session alive holds the SharedContextManager singleton
  // alive through the second compilation. Without it the manager is destroyed and
  // recreated between calls, masking the shared "" context key that is the bug trigger.
  std::unique_ptr<Ort::Session> keep;
  auto buf_a = CompileToEmbedBuffer(model_a, device_name, &keep);
  ASSERT_FALSE(buf_a.empty());

  auto buf_b = CompileToEmbedBuffer(model_b, device_name);
  ASSERT_FALSE(buf_b.empty());

  // Model B is 2×1, model A is 256×256 — buf_b must be much smaller than buf_a.
  // When generation sessions share a context, each new compilation accumulates all
  // prior blobs, so buf_b would be larger than buf_a rather than smaller.
  ASSERT_GT(buf_a.size(), buf_b.size())
      << "buf_b must be smaller than buf_a; embed-mode generation sessions shared a "
         "SharedContext and buf_b accumulated model A's blob.";

  session_options_t options_a(device_name);
  Ort::Session session_a(env_, buf_a.data(), buf_a.size(), options_a);

  session_options_t options_b(device_name);
  Ort::Session session_b(env_, buf_b.data(), buf_b.size(), options_b);

  EXPECT_EQ(RunInference(session_b, dims_b), expected_b)
      << "Session B produced wrong results — buf_b was serialized with model A's blob.";

  EXPECT_EQ(RunInference(session_a, dims_a), expected_a)
      << "Session A results changed after session B was created.";
}
