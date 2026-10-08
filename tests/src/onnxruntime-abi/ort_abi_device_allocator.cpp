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
using io_binding_t = ortAbiTestFixture::io_binding_wrapper_t;
using memory_info_wrapper_t = ortAbiTestFixture::memory_info_wrapper_t;

struct ort_abi_device_allocator : public ortAbiSingleDeviceTestFixture {};

///////////////////////////////////////////////////////////////////////
struct ort_abi_device_allocator_timing
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::tuple<std::string, bool>> {
  std::vector<std::string> additional_required_devices() const override { return {std::get<0>(GetParam())}; }
};

///////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_device_allocator, check_meminfo) {
  device_t device(GetParam());

  auto host_accessible_mi = device.ep_device_.at(0).GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  EXPECT_NE(host_accessible_mi, nullptr);

  auto default_mi = device.ep_device_.at(0).GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  EXPECT_EQ(default_mi, nullptr);  // EP no longer registers a device-typed DEFAULT memory info
}

///////////////////////////////////////////////////////////////////////
// Shared-allocator API (ORT 1.27): exercise the env-level allocator list that
// the default-device-allocators work registers per physical OrtEpDevice
//
// Get/Create/Release round-trip for both device-memory types.
TEST_P(ort_abi_device_allocator, shared_allocator_get_create_release) {
  device_t device(GetParam());
  const auto& ep_device = device.ep_device_.front();

  for (auto mem_type : {OrtDeviceMemoryType_HOST_ACCESSIBLE}) {
    auto mem_info = ep_device.GetMemoryInfo(mem_type);
    ASSERT_NE(static_cast<const OrtMemoryInfo*>(mem_info), nullptr);

    auto created = env_.CreateSharedAllocator(ep_device, mem_type, OrtDeviceAllocator, nullptr);
    EXPECT_NE(static_cast<OrtAllocator*>(created), nullptr);

    auto fetched = env_.GetSharedAllocator(mem_info);
    ASSERT_NE(static_cast<OrtAllocator*>(fetched), nullptr);
    EXPECT_TRUE(fetched.GetInfo().GetAllocatorName().starts_with("OpenVINO_shared_"));

    env_.ReleaseSharedAllocator(ep_device, mem_type);
  }
}

///////////////////////////////////////////////////////////////////////
// Device-memory CopyTensors path — host->device copy, inference on a
// bound device tensor, device->host copy, end-to-end correctness check.
TEST_P(ort_abi_device_allocator, shared_allocator_copy_tensors) {
  device_t device(GetParam());
  const auto& ep_device = device.ep_device_.front();

  constexpr auto mem_type = OrtDeviceMemoryType_HOST_ACCESSIBLE;
  auto device_alloc = env_.CreateSharedAllocator(ep_device, mem_type, OrtDeviceAllocator, nullptr);
  ASSERT_NE(static_cast<OrtAllocator*>(device_alloc), nullptr);

  {
    const auto& model_info = model_library_t::GetModel("mul_1");
    auto input_data = model_info.load_input_data<float>();

    auto session = model_info.create_session(GetParam());
    io_binding_t io_binding(session->ort_session());

    auto cpu_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // host input -> device input (CopyTensor host->device), bind each
    std::vector<Ort::Value> dev_inputs;

    auto& graph = session->graph();
    dev_inputs.reserve(input_data.size());
    for (auto& [name, host_floats] : input_data) {
      const auto& in_desc = graph.input_desc_map_.at(name);
      Ort::Value cpu_in = Ort::Value::CreateTensor<float>(
          cpu_info, host_floats.data(), host_floats.size(), in_desc.shape_.data(), in_desc.shape_.size());
      dev_inputs.push_back(Ort::Value::CreateTensor(
          device_alloc, in_desc.shape_.data(), in_desc.shape_.size(), in_desc.element_type_));
      auto status = env_.CopyTensor(cpu_in, dev_inputs.back(), nullptr);
      ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
      io_binding.bind_input(name, dev_inputs.back());
    }

    // Create device output tensors, bind each
    std::vector<Ort::Value> dev_outputs;
    dev_outputs.reserve(model_info.outputs_.size());
    for (const auto& [name, _] : model_info.outputs_) {
      const auto& out_desc = graph.output_desc_map_.at(name);
      dev_outputs.push_back(Ort::Value::CreateTensor(
          device_alloc, out_desc.shape_.data(), out_desc.shape_.size(), out_desc.element_type_));
      io_binding.bind_output(name, dev_outputs.back());
    }

    graph.run(io_binding);

    // device output -> host (CopyTensor device->host), collect and validate
    io_byte_data_t actual_outputs;
    size_t out_idx = 0;
    for (const auto& [name, _] : model_info.outputs_) {
      const auto& out_desc = graph.output_desc_map_.at(name);
      std::vector<float> host_out(out_desc.size_in_bytes_ / sizeof(float));
      Ort::Value cpu_out = Ort::Value::CreateTensor<float>(
          cpu_info, host_out.data(), host_out.size(), out_desc.shape_.data(), out_desc.shape_.size());
      auto status = env_.CopyTensor(dev_outputs[out_idx++], cpu_out, nullptr);
      ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
      actual_outputs[name] = to_byte_vec(host_out);
    }
    model_info.validate_outputs(actual_outputs);
  }  // all Ort::Values destroyed before release

  env_.ReleaseSharedAllocator(ep_device, mem_type);
}

///////////////////////////////////////////////////////////////////////
// Allocate a buffer larger than each tensor needs via Ort::Allocator, wrap it
// in a shared_ptr that frees it back to the allocator, then build each
// Ort::Value with Value::CreateTensor(info, p_data, ...) pointing at an
// offset into that larger allocation (not the base pointer). Run inference
// end-to-end on the offset-backed tensors to confirm ORT/OpenVINO honor the
// caller-supplied data pointer rather than assuming it is the allocation base.
TEST_P(ort_abi_device_allocator, create_tensor_at_offset) {
  constexpr size_t offset_bytes = 64;

  const auto& model_info = model_library_t::GetModel("mul_1");
  auto input_data = model_info.load_input_data<float>();

  auto session = model_info.create_session(GetParam());
  auto& graph = session->graph();

  auto cpu_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Allocator allocator(session->ort_session(), cpu_info);

  // Keep the base allocations alive for as long as the tensors that point
  // into them are in use; freed back to the allocator on destruction.
  std::vector<std::shared_ptr<void>> backing_allocations;

  auto make_offset_backing = [&](size_t tensor_bytes) {
    void* base_ptr = allocator.Alloc(tensor_bytes + offset_bytes);
    backing_allocations.emplace_back(base_ptr, [&allocator](void* p) { allocator.Free(p); });
    void* tensor_ptr = reinterpret_cast<std::byte*>(base_ptr) + offset_bytes;
    EXPECT_NE(tensor_ptr, base_ptr);
    return tensor_ptr;
  };

  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  for (auto& [name, host_floats] : input_data) {
    const auto& in_desc = graph.input_desc_map_.at(name);
    void* tensor_ptr = make_offset_backing(in_desc.size_in_bytes_);
    std::copy_n(
        reinterpret_cast<const char*>(host_floats.data()),
        in_desc.size_in_bytes_,
        reinterpret_cast<char*>(tensor_ptr));

    input_names.push_back(name.c_str());
    input_tensors.push_back(Ort::Value::CreateTensor(
        cpu_info, tensor_ptr, in_desc.size_in_bytes_, in_desc.shape_.data(), in_desc.shape_.size(), in_desc.element_type_));
  }

  std::vector<const char*> output_names;
  std::vector<Ort::Value> output_tensors;
  for (const auto& [name, _] : model_info.outputs_) {
    const auto& out_desc = graph.output_desc_map_.at(name);
    void* tensor_ptr = make_offset_backing(out_desc.size_in_bytes_);

    output_names.push_back(name.c_str());
    output_tensors.push_back(Ort::Value::CreateTensor(
        cpu_info, tensor_ptr, out_desc.size_in_bytes_, out_desc.shape_.data(), out_desc.shape_.size(), out_desc.element_type_));
  }

  graph.run(input_names, input_tensors, output_names, output_tensors);

  io_byte_data_t actual_outputs;
  size_t out_idx = 0;
  for (const auto& [name, _] : model_info.outputs_) {
    const auto& out_desc = graph.output_desc_map_.at(name);
    std::vector<float> host_out(out_desc.size_in_bytes_ / sizeof(float));
    std::copy_n(
        reinterpret_cast<const char*>(output_tensors[out_idx++].GetTensorRawData()),
        out_desc.size_in_bytes_,
        reinterpret_cast<char*>(host_out.data()));
    actual_outputs[name] = to_byte_vec(host_out);
  }
  model_info.validate_outputs(actual_outputs);
}

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_device_allocator_timing, run) {
  constexpr size_t iterations = 1000;

  device_t device(std::get<0>(GetParam()));
  bool is_io_binding = std::get<1>(GetParam());

  auto time_pipeline = [&](pipeline_t& pipeline) {
    std::vector<ovep_tests::timer_t> timer;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      timer.emplace_back(true);
      pipeline.run();
      timer.back().stop();
    }

    std::vector<double> elapsed;
    std::for_each(
        timer.begin() + 1,
        timer.end(),
        [&](auto& t) { elapsed.push_back(t.elapsed_time_in_milli_secs()); });

    return data_utils::median(elapsed);
  };

  // CPU baseline: allocate tensors from a plain CPU allocator so the comparison
  // below measures the host-accessible device allocator against ordinary host
  // memory
  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  network_t cpu_network{};
  data_utils::read_binary_file("testdata/mul_2_large.onnx", cpu_network.buffer_);
  graph_t cpu_graph(device, cpu_network);
  pipeline_t cpu_pipeline(device, cpu_graph, cpu_mem_info.GetConst());
  auto latency_cpu_allocator = time_pipeline(cpu_pipeline);

  auto run = [&](Ort::ConstMemoryInfo mem_info) {
    network_t network{};
    network.ort_enable_io_binding_ = is_io_binding;
    data_utils::read_binary_file("testdata/mul_2_large.onnx", network.buffer_);

    graph_t graph(device, network);
    pipeline_t pipeline(device, graph, mem_info);

    return time_pipeline(pipeline);
  };

  memory_info_wrapper_t host_accessible_mi{device, OrtDeviceMemoryType_HOST_ACCESSIBLE};
  auto latency_host_accessible = run(host_accessible_mi);
  OVEP_TESTS_LOG_INFO << "latency: cpu_allocator: " << latency_cpu_allocator
                      << " ms, host_accessible: " << latency_host_accessible << " ms";

  if (std::get<0>(GetParam()).find(":") == std::string::npos) {
    // We only expect a difference for non-meta devices. Meta devices don't have a native host accessible allocator.
    EXPECT_LT(latency_host_accessible, latency_cpu_allocator);
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    physical_devices,
    ort_abi_device_allocator,
    ortAbiSingleDeviceTestFixture::all_devices(),
    [](const testing::TestParamInfo<ort_abi_device_allocator::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    direct_tensor_usage,
    ort_abi_device_allocator_timing,
    testing::Values(std::make_tuple("NPU", false), std::make_tuple("GPU", false), std::make_tuple("AUTO:CPU,GPU", false)),
    [](const testing::TestParamInfo<ort_abi_device_allocator_timing::ParamType>& info) {
      std::string result = std::get<0>(info.param);
      std::replace(result.begin(), result.end(), ':', '_');
      std::replace(result.begin(), result.end(), ',', '_');
      return result;
    });

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    io_binding,
    ort_abi_device_allocator_timing,
    testing::Values(std::make_tuple("NPU", true), std::make_tuple("GPU", true), std::make_tuple("AUTO:CPU,GPU", true)),
    [](const testing::TestParamInfo<ort_abi_device_allocator_timing::ParamType>& info) {
      std::string result = std::get<0>(info.param);
      std::replace(result.begin(), result.end(), ':', '_');
      std::replace(result.begin(), result.end(), ',', '_');
      return result;
    });
