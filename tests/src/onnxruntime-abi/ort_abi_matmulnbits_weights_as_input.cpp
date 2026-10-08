// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// Regression gate for the weights-as-input transform on a MatMulNBits model with external B/scales/
// zero_points. The transform must promote the weight B to a graph input while
// leaving the dequantization scale/zero-point as constants. Correctness is validated by inference;
// the transformation itself is checked by dumping the post-transformation OV IR and counting inputs
// (a plain XML scan, so no OpenVINO headers are pulled into the tests).

#include "ortAbiTestFixture.h"
#include "path_utils.h"

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using device_t = ortAbiTestFixture::device_wrapper_t;
using session_options_t = ortAbiTestFixture::session_options_wrapper_t;

namespace {

// Count <layer type="Parameter"> entries in a dumped OV IR .xml. A "dumb" textual scan keeps the
// test free of any OpenVINO dependency.
size_t count_ir_parameters(const fs::path& xml_path) {
  std::ifstream file(xml_path);
  std::string line;
  size_t count = 0;
  while (std::getline(file, line)) {
    if (line.find("type=\"Parameter\"") != std::string::npos) {
      ++count;
    }
  }
  return count;
}

// Find the post-weights-as-input IR dump emitted by the dump_subgraphs provider option.
fs::path find_shared_weights_dump(const fs::path& dir) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto name = entry.path().filename().string();
    if (name.ends_with("_after_transformation_shared_weights.xml")) {
      return entry.path();
    }
  }
  return {};
}

}  // namespace

///////////////////////////////////////////////////////////////////////
struct ort_abi_matmulnbits_weights_as_input : public ortAbiSingleDeviceTestFixture {
  const auto& get_model_info() { return model_library_t::GetModel("matmulnbits_external"); }
};

///////////////////////////////////////////////////////////////////////////
// Run with weight sharing enabled and validate the outputs are unchanged.
TEST_P(ort_abi_matmulnbits_weights_as_input, run_with_weight_sharing) {
  session_options_t session_options{GetParam(), {{kOrtSessionOptionShareEpContexts, "1"}}};
  auto session = get_model_info().create_session(std::move(session_options));
  session->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
// Same, on a model whose zero-point dequant path includes a Slice (unaligned n_blocks_per_col),
// exercising the optional-Slice branch of the dequant exclusion matcher.
TEST_P(ort_abi_matmulnbits_weights_as_input, run_with_weight_sharing_unaligned_zp) {
  session_options_t session_options{GetParam(), {{kOrtSessionOptionShareEpContexts, "1"}}};
  auto session = model_library_t::GetModel("matmulnbits_unaligned").create_session(std::move(session_options));
  session->run_and_validate();
}

///////////////////////////////////////////////////////////////////////////
// Dump the post-transformation IR and assert the weight-as-input transform actually ran: the single
// activation input gains one promoted input for the weight B, while scale and zero-point stay
// constants (i.e. exactly one weight is promoted, not three).
TEST_P(ort_abi_matmulnbits_weights_as_input, weight_promoted_to_input) {
  const auto dump_dir = temp_dir_ / "wai_dump";

  session_options_t session_options{
      GetParam(),
      {{kOrtSessionOptionShareEpContexts, "1"}},
      {{"dump_subgraphs", path_to_utf8(dump_dir)}}};
  auto session = get_model_info().create_session(std::move(session_options));
  session->run_and_validate();

  const auto dump = find_shared_weights_dump(dump_dir);
  ASSERT_FALSE(dump.empty()) << "No post-transformation IR dump found in " << dump_dir;

  // 1 activation input + 1 promoted weight B. Scale and zero-point must remain constants, so the
  // count is 2 rather than 4.
  EXPECT_EQ(count_ir_parameters(dump), 2u) << "Expected only the weight B promoted to an input in " << dump;
}

///////////////////////////////////////////////////////////////////////////
// CPU and NPU only. GPU is excluded: its OpenCL kernel selector cannot lower the u2 type produced by
// the 2-bit MatMulNBits decomposition (a GPU-backend limit, unrelated to weight sharing; u4 is fine).
INSTANTIATE_TEST_SUITE_P(
    ort_abi_matmulnbits_weights_as_input,
    ort_abi_matmulnbits_weights_as_input,
    testing::Values("CPU", "NPU"),
    [](const testing::TestParamInfo<ort_abi_matmulnbits_weights_as_input::ParamType>& info) {
      return info.param;
    });
