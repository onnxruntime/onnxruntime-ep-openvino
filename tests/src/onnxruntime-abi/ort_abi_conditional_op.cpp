// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"

#include <fstream>

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

namespace {
// Parse ORT profiling JSON (Chrome Tracing format) and check if any compute
// node was placed on the given provider. OV EP fuses claimed nodes into a
// single compiled kernel, so we check for provider presence rather than
// individual op_name lookups.
bool has_provider_in_profile(const fs::path& profile_path, const std::string& provider_name) {
  std::ifstream file(profile_path);
  if (!file.is_open()) {
    OVEP_TESTS_LOG_DEBUG << "Profile file could not be opened: " << profile_path;
    return false;
  }

  auto json = nlohmann::json::parse(file, nullptr, false);
  if (json.is_discarded()) {
    OVEP_TESTS_LOG_DEBUG << "Profile file is not valid JSON: " << profile_path;
    return false;
  }

  for (const auto& event : json) {
    if (!event.contains("cat") || event["cat"] != "Node") continue;
    if (!event.contains("args")) continue;
    const auto& args = event["args"];
    OVEP_TESTS_LOG_DEBUG << "Profile node: op_name=" << args.value("op_name", "?")
                         << " provider=" << args.value("provider", "?");
    if (args.contains("provider") && args["provider"] == provider_name) {
      return true;
    }
  }
  return false;
}
}  // namespace

///////////////////////////////////////////////////////////////////////
// Test 1: Simple If op — both branches fully supported by OpenVINO
// then_branch: Relu -> Add(ones)
// else_branch: Sigmoid -> Mul(twos)
// Verifies: session creation and correct branch execution (cond=true)
// Profiling verifies the If node IS placed on OpenVINOEpPlugin.
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_simple
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_simple, run) {
  const auto& model_info = model_library_t::GetModel("if_simple");
  auto session = model_info.create_session(
      GetParam(),
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

TEST_P(ort_abi_if_simple, if_node_claimed_by_ov) {
  const auto& model_info = model_library_t::GetModel("if_simple");
  network_t network = network_t::create(model_info.model_path_, session_model_load_mode_t::file);

  auto profile_prefix = temp_dir_ / "if_simple_profile";
  session_options_wrapper_t session_options(device_wrapper_t(GetParam()), network);
  session_options.EnableProfiling(profile_prefix.native().c_str());

  auto graph = std::make_unique<graph_wrapper_t>(network, std::move(session_options));
  device_wrapper_t device(GetParam());
  auto pipeline = std::make_unique<pipeline_wrapper_t>(device, *graph);

  auto input_data = model_info.load_input_data();
  auto expected_output = model_info.load_expected_output_data();
  io_byte_data_t output_data;
  for (const auto& [name, expected] : expected_output) {
    output_data[name] = std::vector<std::byte>(expected.size());
  }
  pipeline->run(input_data, output_data);
  model_info.validate_outputs(output_data, expected_output);

  Ort::AllocatorWithDefaultOptions allocator;
  auto profile_file = graph->session_->EndProfilingAllocated(allocator);
  fs::path profile_path(profile_file.get());
  ASSERT_TRUE(fs::exists(profile_path)) << "Profile file not found: " << profile_path;

  EXPECT_TRUE(has_provider_in_profile(profile_path, "OpenVINOExecutionProvider"))
      << "OV EP should claim nodes when all branches are supported";

  fs::remove(profile_path);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_simple,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_simple::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Test 2: If op with multiple outputs including a forwarded parent-scope value
// then_branch: Relu -> Add(ones), outputs: [computed, X]
// else_branch: Sigmoid -> Mul(twos), outputs: [computed, X]
// X is forwarded from parent scope via Identity as the second output
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_multi_output_forwarded
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_multi_output_forwarded, run) {
  const auto& model_info = model_library_t::GetModel("if_multi_output_forwarded");
  auto session = model_info.create_session(
      GetParam(),
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_multi_output_forwarded,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_multi_output_forwarded::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Test 3: If op with unsupported custom op in else_branch
// then_branch: Relu -> Add(ones)                              (all supported)
// else_branch: Sigmoid -> IdentityFloat(custom) -> Add(ones)  (custom op NOT in OV)
//
// IdentityFloat is registered in "com.intel.etests" domain and handled by
// ORT's CPU kernel. The If node should NOT be claimed by OV EP.
// With CPU fallback, ORT handles the entire If node.
// Test uses cond=True to take the deterministic Relu path.
// Profiling verifies the If node is NOT placed on OpenVINOEpPlugin.
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_custom_op_branch
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_custom_op_branch, session_creates_with_fallback) {
  const auto& model_info = model_library_t::GetModel("if_custom_op_branch");

  network_t network = network_t::create(model_info.model_path_);
  network.set_cpu_fallback(true);

  auto session = model_info.create_session(GetParam(), std::move(network));
  session->run_and_validate();
}

TEST_P(ort_abi_if_custom_op_branch, if_node_not_claimed_by_ov) {
  const auto& model_info = model_library_t::GetModel("if_custom_op_branch");

  network_t network = network_t::create(model_info.model_path_);
  network.set_cpu_fallback(true);

  auto profile_prefix = temp_dir_ / "if_custom_op_profile";
  session_options_wrapper_t session_options(device_wrapper_t(GetParam()), network);
  session_options.EnableProfiling(profile_prefix.native().c_str());

  auto graph = std::make_unique<graph_wrapper_t>(network, std::move(session_options));
  device_wrapper_t device(GetParam());
  auto pipeline = std::make_unique<pipeline_wrapper_t>(device, *graph);

  auto input_data = model_info.load_input_data();
  auto expected_output = model_info.load_expected_output_data();
  io_byte_data_t output_data;
  for (const auto& [name, expected] : expected_output) {
    output_data[name] = std::vector<std::byte>(expected.size());
  }
  pipeline->run(input_data, output_data);
  model_info.validate_outputs(output_data, expected_output);

  Ort::AllocatorWithDefaultOptions allocator;
  auto profile_file = graph->session_->EndProfilingAllocated(allocator);
  fs::path profile_path(profile_file.get());

  EXPECT_FALSE(has_provider_in_profile(profile_path, "OpenVINOExecutionProvider"))
      << "OV EP should NOT claim nodes when a branch has unsupported custom ops";

  fs::remove(profile_path);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_custom_op_branch,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_custom_op_branch::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Test 4: Nested If op — outer then_branch contains another If node
// outer then: inner If(cond_inner)
//   inner then: Relu -> Add(ones)
//   inner else: Sigmoid -> Mul(twos)
// outer else: Tanh -> Add(threes)
// Verifies: recursive subgraph delegation in ort_graph_delegate
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_nested
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_nested, run) {
  const auto& model_info = model_library_t::GetModel("if_nested");
  auto session = model_info.create_session(
      GetParam(),
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_nested,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_nested::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Test 5: If op with multiple outputs, forwarded X feeds downstream Tanh
// then_branch: Relu -> Add(W), Mul(W), outputs: [A, B, X]
// else_branch: Sigmoid -> Sub(W), Div(W), outputs: [A, B, X]
// X is forwarded from the subgraph as the If node's third output (if_X),
// then consumed by a Tanh node producing graph output "residual".
// Tests passthrough_output_names + downstream consumption of forwarded values.
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_multi_output
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_multi_output, run) {
  const auto& model_info = model_library_t::GetModel("if_multi_output");
  auto session = model_info.create_session(
      GetParam(),
      network_t::create(model_info.model_path_, session_model_load_mode_t::file));
  session->run_and_validate();
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_multi_output,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_multi_output::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
// Test 6: Nested If with unsupported custom op in inner else_branch
// outer then: inner If(cond_inner)
//   inner then: Relu -> Add(ones)                              (supported)
//   inner else: Sigmoid -> IdentityFloat(custom) -> Add(ones)  (custom op NOT in OV)
// outer else: Tanh -> Add(threes)                              (supported)
//
// Tests that unsupported ops deep in nested subgraphs correctly cause
// the outer If node to fall back to CPU EP.
// Profiling verifies the If node is NOT placed on OpenVINOEpPlugin.
///////////////////////////////////////////////////////////////////////
struct ort_abi_if_nested_custom_op
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_if_nested_custom_op, session_creates_with_fallback) {
  const auto& model_info = model_library_t::GetModel("if_nested_custom_op");

  network_t network = network_t::create(model_info.model_path_);
  network.set_cpu_fallback(true);

  auto session = model_info.create_session(GetParam(), std::move(network));
  session->run_and_validate();
}

TEST_P(ort_abi_if_nested_custom_op, if_node_not_claimed_by_ov) {
  const auto& model_info = model_library_t::GetModel("if_nested_custom_op");

  network_t network = network_t::create(model_info.model_path_);
  network.set_cpu_fallback(true);

  auto profile_prefix = temp_dir_ / "if_nested_custom_op_profile";
  session_options_wrapper_t session_options(device_wrapper_t(GetParam()), network);
  session_options.EnableProfiling(profile_prefix.native().c_str());

  auto graph = std::make_unique<graph_wrapper_t>(network, std::move(session_options));
  device_wrapper_t device(GetParam());
  auto pipeline = std::make_unique<pipeline_wrapper_t>(device, *graph);

  auto input_data = model_info.load_input_data();
  auto expected_output = model_info.load_expected_output_data();
  io_byte_data_t output_data;
  for (const auto& [name, expected] : expected_output) {
    output_data[name] = std::vector<std::byte>(expected.size());
  }
  pipeline->run(input_data, output_data);
  model_info.validate_outputs(output_data, expected_output);

  Ort::AllocatorWithDefaultOptions allocator;
  auto profile_file = graph->session_->EndProfilingAllocated(allocator);
  fs::path profile_path(profile_file.get());

  EXPECT_FALSE(has_provider_in_profile(profile_path, "OpenVINOExecutionProvider"))
      << "OV EP should NOT claim nodes when nested subgraph has unsupported custom ops";

  fs::remove(profile_path);
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_conditional_op,
    ort_abi_if_nested_custom_op,
    testing::Values("CPU"),
    [](const testing::TestParamInfo<ort_abi_if_nested_custom_op::ParamType>& info) { return info.param; });
