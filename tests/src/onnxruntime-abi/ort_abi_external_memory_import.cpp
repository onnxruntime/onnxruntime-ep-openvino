// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#ifdef _WIN32

#include "d3d12_utils.h"
#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;
using external_memory_importer_t = ortAbiTestFixture::external_resource_importer_wrapper_t;
using external_memory_t = ortAbiTestFixture::external_memory_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_import_d3d12_heap : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_import_d3d12_heap, run) {
  auto device_name = GetParam();
  device_t device(device_name);

  if (device_name == "GPU") {
    external_memory_importer_t importer(device);
    EXPECT_FALSE(importer.is_valid());
    return;
  }

  external_memory_importer_t importer(device);
  EXPECT_TRUE(importer.can_import_memory(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP));

  auto d3d12_device = d3d12_utils::create_device();

  auto input_heap = d3d12_utils::sys_mem_heap_t{d3d12_device, 4096};
  auto output_heap = d3d12_utils::sys_mem_heap_t{d3d12_device, 4096};

  auto input_handle = input_heap.create_shared_handle();
  auto output_handle = output_heap.create_shared_handle();

  network_t network{};
  network.name_ = fs::path("testdata/mul_1.onnx").string();

  graph_t graph(device, network);

  const auto& input_desc = graph.input_desc_map_.find("X")->second;
  const auto& output_desc = graph.output_desc_map_.find("Y")->second;

  constexpr size_t input_mem_size = 2048;
  constexpr size_t input_heap_offset = 1024;
  constexpr size_t output_mem_size = 2048;
  constexpr size_t output_mem_offset = 1024;

  external_memory_t input_mem(
      importer,
      input_handle.get(),
      input_mem_size,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
      input_heap_offset);
  external_memory_t output_mem(importer, output_handle.get(), output_mem_size);

  EXPECT_THROW(input_mem.create_tensor(std::vector<int64_t>{1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
               std::exception);

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(input_mem.create_tensor(input_desc.shape_, input_desc.element_type_));

  std::vector<Ort::Value> output_tensors;
  output_tensors.emplace_back(
      output_mem.create_tensor(output_desc.shape_, output_desc.element_type_, output_mem_offset));

  input_heap.fill(std::vector<float>(6, 2.f), 1024);

  graph.run(graph.input_names(), input_tensors, graph.output_names(), output_tensors);

  EXPECT_THAT(output_heap.get<float>(6, 1024), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_external_memory_import,
    ort_abi_import_d3d12_heap,
    testing::Values("NPU", "GPU"),
    [](const testing::TestParamInfo<ort_abi_import_d3d12_heap::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_import_d3d12_buffer : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_import_d3d12_buffer, run) {
  auto device_name = GetParam();
  device_t device(device_name);

  if (device_name == "GPU") {
    external_memory_importer_t importer(device);
    EXPECT_FALSE(importer.is_valid());
    return;
  }

  external_memory_importer_t importer(device);

  auto d3d12_device = d3d12_utils::create_device();
  auto buffer = d3d12_utils::sys_mem_heap_t{d3d12_device, 4096};
  auto handle = buffer.create_shared_handle();

  constexpr size_t size = 2048;

  EXPECT_FALSE(importer.can_import_memory(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE));
  EXPECT_THROW(
      external_memory_t(importer, handle.get(), size, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE),
      std::exception);
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_external_memory_import,
    ort_abi_import_d3d12_buffer,
    testing::Values("NPU", "GPU"),
    [](const testing::TestParamInfo<ort_abi_import_d3d12_buffer::ParamType>& info) { return info.param; });

#endif  // _WIN32
