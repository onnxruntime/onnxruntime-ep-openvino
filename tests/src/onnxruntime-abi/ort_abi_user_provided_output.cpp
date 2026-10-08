// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ortAbiTestFixture.h"
#include "ort_test_version.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using model_info_t = ortAbiTestFixture::model_library_t::model_info_t;
using builtin_session_t = ortAbiTestFixture::builtin_session_t;

namespace {

constexpr std::array<int64_t, 2> kMulShape{3, 2};
constexpr std::array<int64_t, 2> kSwappedMulShape{2, 3};

struct MulTestContext {
  const model_info_t& model_info;
  std::unique_ptr<builtin_session_t> session;
  Ort::MemoryInfo memory_info;
  std::string input_name;
  std::string output_name;
  std::vector<float> input_data;
  std::vector<float> expected_output;

  MulTestContext()
      : model_info(ortAbiTestFixture::model_library_t::GetModel("mul_1_dynamic")),
        session(model_info.create_session("CPU")),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_name(session->ort_session().get_tensor_name(0, true)),
        output_name(session->ort_session().get_tensor_name(0, false)) {
    auto input_data_by_name = model_info.load_input_data<float>();
    input_data = std::move(input_data_by_name.at(input_name));

    auto expected_output_by_name = model_info.load_expected_output_data<float>();
    expected_output = std::move(expected_output_by_name.at(output_name));
  }

  std::vector<float> scaled_input(float scale) const {
    auto result = input_data;
    for (auto& value : result) {
      value *= scale;
    }
    return result;
  }

  std::vector<float> scaled_output(float scale) const {
    auto result = expected_output;
    for (auto& value : result) {
      value *= scale;
    }
    return result;
  }

  bool has_dynamic_input_shape() const {
    const auto shape = session->ort_session().get_type_info(0, true).GetTensorTypeAndShapeInfo().GetShape();
    for (const auto dimension : shape) {
      if (dimension < 0) {
        return true;
      }
    }
    return false;
  }

  Ort::Value make_input(std::vector<float>& data) {
    return Ort::Value::CreateTensor<float>(
        memory_info, data.data(), data.size(), kMulShape.data(), kMulShape.size());
  }

  Ort::Value make_output(std::vector<float>& data) { return make_output(data, kMulShape); }

  Ort::Value make_output(std::vector<float>& data, const std::array<int64_t, 2>& shape) {
    return Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
  }

  std::vector<Ort::Value> run(Ort::Value input, Ort::Value output) {
    std::vector<Ort::Value> outputs;
    outputs.emplace_back(std::move(output));
    return run(std::move(input), std::move(outputs));
  }

  std::vector<Ort::Value> run_with_unallocated_output(std::vector<float>& input) {
    std::vector<Ort::Value> outputs;
    outputs.emplace_back(nullptr);
    return run(make_input(input), std::move(outputs));
  }

 private:
  std::vector<Ort::Value> run(Ort::Value input, std::vector<Ort::Value> outputs) {
    const std::vector<const char*> input_names = {input_name.c_str()};
    const std::vector<const char*> output_names = {output_name.c_str()};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input));
    session->ort_session().run(input_names, inputs, output_names, outputs);
    return outputs;
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////
struct ort_abi_user_provided_output : public ortAbiTestFixture {};

TEST_F(ort_abi_user_provided_output, run_outputs_rebind_and_fallback_are_isolated) {
  OVEP_SKIP_IF_ORT_API_BELOW(30);

  const auto expect_output = [](const std::vector<Ort::Value>& outputs, const std::vector<float>& expected) {
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());
    ASSERT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetElementCount(), expected.size());
    const auto* data = outputs[0].GetTensorData<float>();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(std::vector<float>(data, data + expected.size()), expected);
  };

  // Use different inputs on each invocation. If a fallback request retains a
  // caller buffer from a previous direct invocation, the distinct result will
  // expose that stale binding.
  MulTestContext test;
  ASSERT_TRUE(test.has_dynamic_input_shape());
  auto fallback_input = test.scaled_input(1.0f);
  const auto fallback_expected = test.scaled_output(1.0f);
  auto fallback_outputs = test.run_with_unallocated_output(fallback_input);
  expect_output(fallback_outputs, fallback_expected);

  auto first_input = test.scaled_input(2.0f);
  const auto first_expected = test.scaled_output(2.0f);
  std::vector<float> first_output(first_expected.size(), -1.0f);
  auto direct_outputs = test.run(test.make_input(first_input), test.make_output(first_output));
  ASSERT_EQ(direct_outputs.size(), 1u);
  EXPECT_EQ(direct_outputs[0].GetTensorMutableRawData(), first_output.data());
  EXPECT_EQ(first_output, first_expected);

  // A second direct call must overwrite every output binding with the new
  // caller buffer rather than reusing the first call's pointer.
  auto second_input = test.scaled_input(3.0f);
  const auto second_expected = test.scaled_output(3.0f);
  std::vector<float> second_output(second_expected.size(), -2.0f);
  auto second_outputs = test.run(test.make_input(second_input), test.make_output(second_output));
  ASSERT_EQ(second_outputs.size(), 1u);
  EXPECT_EQ(second_outputs[0].GetTensorMutableRawData(), second_output.data());
  EXPECT_EQ(second_output, second_expected);

  // Alternate back to the fallback path. The dynamic pool must remain
  // pristine and the previous direct buffers must not be touched.
  auto final_input = test.scaled_input(4.0f);
  const auto final_expected = test.scaled_output(4.0f);
  auto final_outputs = test.run_with_unallocated_output(final_input);
  expect_output(final_outputs, final_expected);
  const auto* final_data = final_outputs[0].GetTensorData<float>();
  ASSERT_NE(final_data, nullptr);
  EXPECT_NE(final_data, first_output.data());
  EXPECT_NE(final_data, second_output.data());
  EXPECT_EQ(first_output, first_expected);
  EXPECT_EQ(second_output, second_expected);
}

TEST_F(ort_abi_user_provided_output, equal_capacity_shape_mismatch_fails) {
  OVEP_SKIP_IF_ORT_API_BELOW(30);

  MulTestContext test;
  auto input_data = test.scaled_input(1.0f);
  std::vector<float> output_data(test.expected_output.size(), -1.0f);

  try {
    static_cast<void>(test.run(test.make_input(input_data), test.make_output(output_data, kSwappedMulShape)));
    FAIL() << "A user-provided output with the wrong shape must fail the Run";
  } catch (const Ort::Exception& ex) {
    EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
  }
}
