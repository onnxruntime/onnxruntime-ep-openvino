// Copyright (C) Intel Corporation
// Licensed under the MIT License

#ifdef _WIN32

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;
using external_memory_importer_t = ortAbiTestFixture::external_resource_importer_wrapper_t;
using external_memory_t = ortAbiTestFixture::external_memory_wrapper_t;

namespace {

///////////////////////////////////////////////////////////////////////
// Windows-only page-aligned allocation helper for CPU_VA import tests.
struct aligned_buffer_t {
  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(aligned_buffer_t);

  explicit aligned_buffer_t(size_t size) : size_(size) {
    ptr_ = _aligned_malloc(size, 4096);
    if (ptr_ == nullptr) {
      OVEP_TESTS_THROW("Failed to allocate page-aligned buffer");
    }
    std::memset(ptr_, 0, size_);
  }

  ~aligned_buffer_t() {
    if (ptr_) {
      _aligned_free(ptr_);
      ptr_ = nullptr;
    }
  }

  void* get() const { return ptr_; }
  size_t size() const { return size_; }

  template <typename T>
  void fill(const std::vector<T>& data, size_t offset_bytes = 0) {
    std::memcpy(static_cast<uint8_t*>(ptr_) + offset_bytes, data.data(), data.size() * sizeof(T));
  }

  template <typename T>
  std::vector<T> get(size_t count, size_t offset_bytes = 0) const {
    std::vector<T> result(count);
    std::memcpy(result.data(), static_cast<const uint8_t*>(ptr_) + offset_bytes, count * sizeof(T));
    return result;
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

}  // namespace

///////////////////////////////////////////////////////////////////////
struct ort_abi_import_cpu_va
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_import_cpu_va, null_importer_for_cpu_and_gpu) {
  auto device_name = GetParam();
  device_t device(device_name);
  if (device.ep_device_.empty()) {
    GTEST_SKIP() << device_name << " not available on this machine.";
  }

  external_memory_importer_t importer(device);

  if (device_name == "NPU") {
    EXPECT_TRUE(importer.is_valid());
  } else {
    EXPECT_FALSE(importer.is_valid());
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_import_cpu_va,
    ort_abi_import_cpu_va,
    testing::Values("CPU", "GPU", "NPU"),
    [](const testing::TestParamInfo<ort_abi_import_cpu_va::ParamType>& info) { return info.param; });

///////////////////////////////////////////////////////////////////////
struct ort_abi_import_cpu_va_npu : public ortAbiTestFixture {
  void SetUp() override {
    ortAbiTestFixture::SetUp();
    npu_ = device_t("NPU");
    if (npu_.ep_device_.empty()) {
      GTEST_SKIP() << "NPU not available on this machine.";
    }
    importer_ = std::make_unique<external_memory_importer_t>(npu_);
    ASSERT_TRUE(importer_->is_valid());
  }

  void TearDown() override {
    importer_.reset();
    ortAbiTestFixture::TearDown();
  }

  device_t npu_;
  std::unique_ptr<external_memory_importer_t> importer_;
};

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, unsupported_handle_type_returns_not_implemented) {
  external_memory_importer_t& importer = *importer_;

  aligned_buffer_t buffer(4096);

  OrtExternalMemoryDescriptor desc{
      .version = ORT_API_VERSION,
      .handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32,
      .native_handle = buffer.get(),
      .size_bytes = buffer.size(),
      .offset_bytes = 0,
  };

  OrtExternalMemoryHandle* handle = nullptr;
  Ort::Status status{Ort::GetInteropApi().ImportMemory(importer.get(), &desc, &handle)};
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
}

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, import_page_aligned_buffer) {
  external_memory_importer_t& importer = *importer_;
  EXPECT_TRUE(importer.can_import_memory(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));

  aligned_buffer_t buffer(4096);

  external_memory_t mem(importer, buffer.get(), buffer.size(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);

  std::vector<int64_t> shape{static_cast<int64_t>(buffer.size())};
  Ort::Value tensor{mem.create_tensor(shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};

  const uint8_t* data = tensor.GetTensorData<uint8_t>();
  EXPECT_EQ(static_cast<const void*>(data), buffer.get());
}

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, unaligned_address_rejected) {
  external_memory_importer_t& importer = *importer_;

  aligned_buffer_t buffer(4096);
  void* unaligned = static_cast<uint8_t*>(buffer.get()) + 1;

  OrtExternalMemoryDescriptor desc{
      .version = ORT_API_VERSION,
      .handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION,
      .native_handle = unaligned,
      .size_bytes = 100,
      .offset_bytes = 0,
  };

  OrtExternalMemoryHandle* handle = nullptr;
  Ort::Status status{Ort::GetInteropApi().ImportMemory(importer.get(), &desc, &handle)};
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
}

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, unaligned_size_accepted) {
  external_memory_importer_t& importer = *importer_;

  aligned_buffer_t buffer(4096);

  external_memory_t mem(importer, buffer.get(), 100, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);

  std::vector<int64_t> shape{100};
  Ort::Value tensor{mem.create_tensor(shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};

  EXPECT_EQ(tensor.GetTensorTypeAndShapeInfo().GetElementCount(), 100u);
}

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, inference_end_to_end_is_zero_copy) {
  external_memory_importer_t& importer = *importer_;

  network_t network{};
  network.name_ = fs::path("testdata/mul_1.onnx").string();

  graph_t graph(npu_, network);

  const auto& input_desc = graph.input_desc_map_.find("X")->second;
  const auto& output_desc = graph.output_desc_map_.find("Y")->second;

  constexpr size_t input_mem_size = 4096;
  constexpr size_t output_mem_size = 4096;

  aligned_buffer_t input_buffer(input_mem_size);
  aligned_buffer_t output_buffer(output_mem_size);

  external_memory_t input_mem(importer, input_buffer.get(), input_mem_size, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);
  external_memory_t output_mem(importer, output_buffer.get(), output_mem_size, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(input_mem.create_tensor(input_desc.shape_, input_desc.element_type_));

  std::vector<Ort::Value> output_tensors;
  output_tensors.emplace_back(output_mem.create_tensor(output_desc.shape_, output_desc.element_type_));

  // Write the sentinel input after create_tensor but before Run to prove the tensor is a live
  // zero-copy view over input_buffer, not a snapshot taken at import/create_tensor time.
  input_buffer.fill(std::vector<float>(6, 2.f));

  graph.run(graph.input_names(), input_tensors, graph.output_names(), output_tensors);

  EXPECT_THAT(output_buffer.get<float>(6), ::testing::ElementsAre(2.f, 4.f, 6.f, 8.f, 10.f, 12.f));
}

///////////////////////////////////////////////////////////////////////////
// The L0 driver on this hardware does not reject a second import of the same CPU_VA region: it
// hands back an independent handle wrapping the same underlying memory. Both handles must be
// independently usable and independently releasable without crashing or corrupting the region.
TEST_F(ort_abi_import_cpu_va_npu, double_import_same_region_allowed) {
  external_memory_importer_t& importer = *importer_;

  aligned_buffer_t buffer(4096);

  external_memory_t first(importer, buffer.get(), buffer.size(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);
  external_memory_t second(importer, buffer.get(), buffer.size(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);

  EXPECT_NE(first.get(), nullptr);
  EXPECT_NE(second.get(), nullptr);
  EXPECT_NE(first.get(), second.get());

  std::vector<int64_t> shape{static_cast<int64_t>(buffer.size())};
  Ort::Value first_tensor{first.create_tensor(shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};
  Ort::Value second_tensor{second.create_tensor(shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)};

  EXPECT_EQ(static_cast<const void*>(first_tensor.GetTensorData<uint8_t>()), buffer.get());
  EXPECT_EQ(static_cast<const void*>(second_tensor.GetTensorData<uint8_t>()), buffer.get());
}

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_import_cpu_va_npu, handle_release_frees_l0_memory) {
  external_memory_importer_t& importer = *importer_;

  aligned_buffer_t buffer(4096);

  {
    external_memory_t mem(importer, buffer.get(), buffer.size(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);
  }

  // Sanity check that Release() doesn't leave the importer or L0 context in a broken state:
  // importing the same region again after the first handle is released must still succeed.
  // Note this hardware's driver allows overlapping imports of the same region even without an
  // intervening Release (see double_import_same_region_allowed), so this alone can't detect a
  // leaked ZeroMem pool entry -- it only guards against Release corrupting importer state.
  external_memory_t reimported(importer, buffer.get(), buffer.size(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION);
  EXPECT_NE(reimported.get(), nullptr);
}

#endif  // _WIN32
