// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;
using model_library_t = ortAbiTestFixture::model_library_t;
using builtin_session_t = ortAbiTestFixture::builtin_session_t;

namespace {
constexpr uint8_t kXorKey = 0xA5;

// "OVEP_BIN" header magic, must match common/ov_bin_format.h.
constexpr std::array<uint8_t, 8> kExpectedMagicBytes = {'O', 'V', 'E', 'P', '_', 'B', 'I', 'N'};

void xor_in_place(void* buffer, size_t size) {
  auto* bytes = static_cast<uint8_t*>(buffer);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] ^= kXorKey;
  }
}

// A trivial "encrypt-on-write / decrypt-on-read" host store used to validate the v28 EPContext
// data callbacks. The write callback stashes XOR-encrypted bytes into an in-memory map; the read
// callback fetches them back and XOR-decrypts before handing the buffer to the EP.
struct callback_state_t {
  std::mutex mutex;
  std::map<std::string, std::vector<uint8_t> > blobs;
  std::atomic<size_t> write_calls{0};
  std::atomic<size_t> read_calls{0};
};

OrtStatus* ORT_API_CALL xor_write_cb(void* state_, const char* file_name, const void* buffer, size_t buffer_num_bytes) {
  auto* state = static_cast<callback_state_t*>(state_);
  state->write_calls.fetch_add(1, std::memory_order_relaxed);

  std::vector<uint8_t> encrypted(buffer_num_bytes);
  std::memcpy(encrypted.data(), buffer, buffer_num_bytes);
  xor_in_place(encrypted.data(), encrypted.size());

  std::lock_guard<std::mutex> lock(state->mutex);
  state->blobs[file_name] = std::move(encrypted);
  return nullptr;
}

OrtStatus* ORT_API_CALL xor_read_cb(
    void* state_,
    const char* file_name,
    OrtAllocator* allocator,
    void** buffer,
    size_t* data_size) {
  auto* state = static_cast<callback_state_t*>(state_);
  state->read_calls.fetch_add(1, std::memory_order_relaxed);

  std::vector<uint8_t> copy;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto it = state->blobs.find(file_name);
    if (it == state->blobs.end()) {
      return Ort::GetApi().CreateStatus(
          ORT_FAIL,
          (std::string("blob not found in callback state: ") + file_name).c_str());
    }
    copy = it->second;
  }

  void* out = allocator->Alloc(allocator, copy.size());
  if (out == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "allocator returned null");
  }
  std::memcpy(out, copy.data(), copy.size());
  xor_in_place(out, copy.size());

  *buffer = out;
  *data_size = copy.size();
  return nullptr;
}

}  // namespace

///////////////////////////////////////////////////////////////////////
// NOTE: Only the "no sharing" round trip is exercised here. A share+stop variant cannot use
// the v28 write callback today because that callback is exposed only on
// `OrtModelCompilationOptions`, which means generation happens through `Ort::CompileModel`.
// Each `CompileModel` call destroys its EP plugin instance after returning, taking the
// `SharedContextManager` `WeakSingleton` down with it, so two CompileModel calls cannot
// accumulate partitions in a single in-memory blob. Extending lifetime across CompileModel
// invocations is out of scope for this change.
struct ort_abi_ep_context_callbacks
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_ep_context_callbacks, no_share_external_round_trip) {
  const auto& ov_device_name = GetParam();

  const auto& model_info = model_library_t::GetModel("mul_1");

  const fs::path output_epctx_model = temp_dir_ / ("mul_1_ctx_cb_" + ov_device_name + ".onnx");

  callback_state_t state;

  // Generate EPContext via Ort::CompileModel, delegating the bin write to our callback.
  {
    session_options_t session_options(ov_device_name);
    Ort::ModelCompilationOptions compile_options(env_, session_options);
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetInputModelPath(model_info.model_path_.native().c_str());
    compile_options.SetOutputModelPath(output_epctx_model.native().c_str());

    auto* set_write_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(&Ort::GetApi());
    Ort::ThrowOnError(set_write_fn(compile_options, xor_write_cb, &state));

    Ort::CompileModel(env_, compile_options);
  }

  EXPECT_TRUE(fs::exists(output_epctx_model));
  EXPECT_EQ(state.write_calls.load(), 1u);
  ASSERT_EQ(state.blobs.size(), 1u);

  // The EP must not have spilled the blob to disk alongside the EPContext ONNX.
  const fs::path implicit_bin_path =
      output_epctx_model.parent_path() / (output_epctx_model.stem().string() + "_OpenVINOExecutionProvider.bin");
  EXPECT_FALSE(fs::exists(implicit_bin_path))
      << "EP should not have written bin to disk when write callback is set";

  // XOR-decrypting the captured bytes must reveal the OVEP_BIN header magic; confirms encryption
  // round-trips end-to-end and rules out the EP accidentally writing the blob in cleartext.
  {
    const auto& encrypted = state.blobs.begin()->second;
    ASSERT_GE(encrypted.size(), kExpectedMagicBytes.size());
    for (size_t i = 0; i < kExpectedMagicBytes.size(); ++i) {
      EXPECT_EQ(static_cast<uint8_t>(encrypted[i] ^ kXorKey), kExpectedMagicBytes[i])
          << "byte " << i << " mismatch - encryption may not have been applied end-to-end";
    }
  }

  // Load the produced EPContext model with the read callback registered, and validate inference.
  // The session must be created FROM the EPContext ONNX (not from the original model with a hint)
  // so that ORT presents the EPContext nodes directly to the EP, which in turn triggers the read
  // callback path in InitializeSharedContext.
  {
    session_options_t session_options(ov_device_name);
    auto* set_read_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&Ort::GetApi());
    Ort::ThrowOnError(set_read_fn(session_options, xor_read_cb, &state));

    // Redirect model_path_ to the EPContext ONNX so builtin_session_t loads from it.
    auto epctx_model_info = model_info;
    epctx_model_info.model_path_ = output_epctx_model;

    auto session = epctx_model_info.create_session(std::move(session_options));
    session->run_and_validate();
  }

  EXPECT_GE(state.read_calls.load(), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context_callbacks,
    ort_abi_ep_context_callbacks,
    testing::Values(std::string("CPU")),
    [](const testing::TestParamInfo<ort_abi_ep_context_callbacks::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Regression test: BinManager::Deserialize(shared_context) must be idempotent when two
// inference sessions load the same external EPContext bin path simultaneously.
//
// Generation uses the share+stop pattern. Two inference sessions are then created while both
// are alive so SharedContextManager::GetOrCreateSharedContext returns the same SharedContext
// to both — the second Deserialize() call must be a silent no-op, not a crash or clobber.
struct ort_abi_ep_ctx_shared_bin_reuse
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_ep_ctx_shared_bin_reuse, two_concurrent_sessions_from_same_bin) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("mul_1");
  const fs::path epctx_model = temp_dir_ / ("mul_1_ctx_shared_" + ov_device_name + ".onnx");

  // Phase 1: Share+stop generation using a single session that both enables sharing (uses
  // GetOrCreateActiveSharedContext) and finalizes (stop=1 → Serialize() writes the bin).
  // Using share=1+stop=1 together in one session avoids the "file already exists" error that
  // occurs when two separate sessions write EPContext ONNX to the same path.
  {
    session_options_t gen_opts(ov_device_name,
                               {{kOrtSessionOptionEpContextEnable, "1"},
                                {kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model)},
                                {kOrtSessionOptionEpContextEmbedMode, "0"},
                                {kOrtSessionOptionShareEpContexts, "1"},
                                {kOrtSessionOptionStopShareEpContexts, "1"}});
    model_info.create_session(std::move(gen_opts));
  }

  ASSERT_TRUE(fs::exists(epctx_model)) << "EPContext not created: " << path_to_utf8(epctx_model);

  // Phase 2: Two inference sessions created while both are alive.
  // Load directly from the EPContext ONNX so ORT presents EPContext nodes to the EP.
  // session1 keeps the SharedContextManager WeakSingleton alive while session2 is created,
  // so GetOrCreateSharedContext returns the same SharedContext to both.
  // The second BinManager::Deserialize(shared_context) call must be an idempotent no-op.
  auto epctx_model_info = model_info;
  epctx_model_info.model_path_ = epctx_model;

  auto session1 = epctx_model_info.create_session(session_options_t(ov_device_name));
  auto session2 = epctx_model_info.create_session(session_options_t(ov_device_name));

  // Both sessions must produce correct inference results — if the shared compiled-model state
  // were corrupted, run_and_validate would fail with wrong outputs or a crash.
  session1->run_and_validate();
  session2->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context_callbacks,
    ort_abi_ep_ctx_shared_bin_reuse,
    testing::Values(std::string("CPU")),
    [](const testing::TestParamInfo<ort_abi_ep_ctx_shared_bin_reuse::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Regression test: two distinct models that resolve to the same bin path must each load their own
// blob through the read callback while both sessions are alive. The SharedContextManager cache is
// keyed by the full resolved bin path (epctx_dir / ep_cache_context), so we compile the same model
// twice under one stem and load both EPContext ONNX files from a single directory to force that
// path to collide. Before the fix, the second load reused the first session's already-deserialized
// context and GetNativeBlob() missed its partition ("Blob not found"); the fix gives each
// read-callback session a private SharedContext.
struct ort_abi_ep_ctx_distinct_models_same_bin
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_ep_ctx_distinct_models_same_bin, concurrent_load_serves_each_own_blob) {
  const auto& ov_device_name = GetParam();
  const auto& model_info = model_library_t::GetModel("mul_1");

  // Same stem in two compile dirs -> identical bin file name, distinct EPContext ONNX on disk.
  const fs::path dir_a = temp_dir_ / "collide_a";
  const fs::path dir_b = temp_dir_ / "collide_b";
  fs::create_directories(dir_a);
  fs::create_directories(dir_b);
  const fs::path epctx_a = dir_a / "collide.onnx";
  const fs::path epctx_b = dir_b / "collide.onnx";

  callback_state_t state_a;
  callback_state_t state_b;

  auto compile_to = [&](const fs::path& out_model, callback_state_t& state) {
    session_options_t session_options(ov_device_name);
    Ort::ModelCompilationOptions compile_options(env_, session_options);
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetInputModelPath(model_info.model_path_.native().c_str());
    compile_options.SetOutputModelPath(out_model.native().c_str());

    auto* set_write_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(&Ort::GetApi());
    Ort::ThrowOnError(set_write_fn(compile_options, xor_write_cb, &state));

    Ort::CompileModel(env_, compile_options);
  };

  compile_to(epctx_a, state_a);
  compile_to(epctx_b, state_b);

  // Precondition: both models must have captured a blob under the SAME (colliding) bin key.
  ASSERT_EQ(state_a.blobs.size(), 1u);
  ASSERT_EQ(state_b.blobs.size(), 1u);
  ASSERT_EQ(state_a.blobs.begin()->first, state_b.blobs.begin()->first)
      << "test precondition: both models must resolve to the same bin file name";

  // Load both from ONE directory so their full bin paths (epctx_dir / ep_cache_context) collide --
  // this is what forces the cross-session cache hit; different directories give each its own entry.
  // The .bin is never on disk (captured in state_*), so only the ONNX files need copying.
  const fs::path load_dir = temp_dir_ / "collide_load";
  fs::create_directories(load_dir);
  const fs::path load_a = load_dir / "model_a.onnx";
  const fs::path load_b = load_dir / "model_b.onnx";
  fs::copy_file(epctx_a, load_a);
  fs::copy_file(epctx_b, load_b);

  // create_session holds a *reference* to the model_info, so each copy must outlive its session --
  // keep them in test-body scope, not local to the load lambda (that would dangle after it returns).
  auto model_info_a = model_info;
  model_info_a.model_path_ = load_a;
  auto model_info_b = model_info;
  model_info_b.model_path_ = load_b;

  auto load = [&](const auto& epctx_model_info, callback_state_t& state) {
    session_options_t session_options(ov_device_name);
    auto* set_read_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&Ort::GetApi());
    Ort::ThrowOnError(set_read_fn(session_options, xor_read_cb, &state));

    return epctx_model_info.create_session(std::move(session_options));
  };

  // Keep session1 alive (holds the SharedContextManager WeakSingleton) while creating session2,
  // so a path-keyed cache would hand session2 session1's context.
  auto session1 = load(model_info_a, state_a);
  auto session2 = load(model_info_b, state_b);

  session1->run_and_validate();
  session2->run_and_validate();

  EXPECT_GE(state_a.read_calls.load(), 1u);
  EXPECT_GE(state_b.read_calls.load(), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_context_callbacks,
    ort_abi_ep_ctx_distinct_models_same_bin,
    testing::Values(std::string("CPU")),
    [](const testing::TestParamInfo<ort_abi_ep_ctx_distinct_models_same_bin::ParamType>& info) { return info.param; });
