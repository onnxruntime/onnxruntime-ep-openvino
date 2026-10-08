// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

#include <array>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;

///////////////////////////////////////////////////////////////////////
// Exercises the ov_compute.cc dynamic-IO string-output fix: Reshape on a
// STRING tensor with an unranked/symbolic output shape, forcing the
// element-wise FillStringTensorElement path instead of memcpy.
struct ort_abi_string_tensor_reshape : public ortAbiTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_string_tensor_reshape, dynamic_shape_flatten) {
  device_t device("CPU");

  network_t network{};
  data_utils::read_binary_file("testdata/reshape_string.onnx", network.buffer_);
  network.set_cpu_fallback(false);

  graph_t graph(device, network);

  Ort::AllocatorWithDefaultOptions allocator;

  const std::vector<std::string> prompt_strings = {"hello", "world", "openvino"};
  const std::array<int64_t, 1> input_shape{static_cast<int64_t>(prompt_strings.size())};

  auto input_tensor = Ort::Value::CreateTensor(allocator, input_shape.data(), input_shape.size(),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  std::vector<const char*> raw_strings;
  raw_strings.reserve(prompt_strings.size());
  for (const auto& s : prompt_strings) {
    raw_strings.push_back(s.c_str());
  }
  input_tensor.FillStringTensor(raw_strings.data(), raw_strings.size());

  std::vector<const char*> input_names = {"prompt"};
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor));

  // Default-constructed placeholder: ORT allocates and fills in a fresh,
  // dynamically-shaped OrtValue in place since the output shape is unranked.
  std::vector<const char*> output_names = {"flattened"};
  std::vector<Ort::Value> output_tensors;
  output_tensors.emplace_back(nullptr);

  graph.run(input_names, input_tensors, output_names, output_tensors);

  ASSERT_EQ(output_tensors.size(), 1u);
  const auto& output = output_tensors[0];
  const size_t element_count = output.GetTensorTypeAndShapeInfo().GetElementCount();
  ASSERT_EQ(element_count, prompt_strings.size());

  for (size_t i = 0; i < element_count; ++i) {
    EXPECT_EQ(output.GetStringTensorElement(i), prompt_strings[i]);
  }
}
