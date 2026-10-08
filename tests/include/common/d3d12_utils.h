// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32

#include <d3d12.h>
#include <wrl/client.h>
#include <stdexcept>
#include <dxgi.h>
#include <dxgi1_4.h>

template <typename T>
using com_ptr = Microsoft::WRL::ComPtr<T>;

namespace ovep_tests {
namespace d3d12_utils {

///////////////////////////////////////////////////////////////////////
struct sys_mem_heap_t {
  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(sys_mem_heap_t);

  ///////////////////////////////////////////////////////////////////////
  sys_mem_heap_t(com_ptr<ID3D12Device3> device, size_t alloc_size) : device_(device), size_(alloc_size) {
    mem_ = VirtualAlloc(nullptr, size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (mem_ == nullptr) {
      OVEP_TESTS_THROW("Failed to allocate system memory");
    }

    if (FAILED(device_->OpenExistingHeapFromAddress(mem_, IID_PPV_ARGS(&ptr_)))) {
      VirtualFree(mem_, 0, MEM_RELEASE);
      mem_ = nullptr;
      OVEP_TESTS_THROW("Failed to create system memory Heap");
    }
  }

  ///////////////////////////////////////////////////////////////////////
  ~sys_mem_heap_t() {
    ptr_.Reset();
    if (mem_) {
      VirtualFree(mem_, 0, MEM_RELEASE);
      mem_ = nullptr;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  std::unique_ptr<void, void (*)(HANDLE)> create_shared_handle() const {
    HANDLE shared_handle = nullptr;
    if (FAILED(device_->CreateSharedHandle(ptr_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle))) {
      OVEP_TESTS_THROW("Failed to create shared handle from Heap");
    }
    return std::unique_ptr<void, void (*)(HANDLE)>(
        shared_handle,
        [](HANDLE h) {
          if (h) {
            CloseHandle(h);
          }
        });
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T>
  void fill(const std::vector<T>& data, size_t offset_bytes = 0) {
    size_t data_size_bytes = data.size() * sizeof(T);
    if (offset_bytes + data_size_bytes > size_) {
      OVEP_TESTS_THROW("Fill operation exceeds heap size");
    }

    void* dest = static_cast<uint8_t*>(mem_) + offset_bytes;
    std::memcpy(dest, data.data(), data_size_bytes);
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T>
  std::vector<T> get(size_t count, size_t offset_bytes = 0) const {
    size_t data_size_bytes = count * sizeof(T);
    if (offset_bytes + data_size_bytes > size_) {
      OVEP_TESTS_THROW("Get operation exceeds heap size");
    }

    std::vector<T> result(count);
    const void* src = static_cast<const uint8_t*>(mem_) + offset_bytes;
    std::memcpy(result.data(), src, data_size_bytes);

    return result;
  }

  ///////////////////////////////////////////////////////////////////////
  com_ptr<ID3D12Device3> device() const { return device_; }

  ///////////////////////////////////////////////////////////////////////
  auto create_buffer(size_t buffer_size, size_t heap_offset) const {
    D3D12_RESOURCE_DESC desc{
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = buffer_size,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc =
            {
                .Count = 1,
                .Quality = 0,
            },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
    };

    com_ptr<ID3D12Resource> resource;
    if (FAILED(device_->CreatePlacedResource(
            ptr_.Get(),
            heap_offset,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&resource)))) {
      OVEP_TESTS_THROW("Failed to create placed resource on heap");
    }

    return resource;
  }

 private:
  com_ptr<ID3D12Device3> device_;
  com_ptr<ID3D12Heap> ptr_;
  void* mem_ = nullptr;
  size_t size_ = 0;
};

///////////////////////////////////////////////////////////////////////
struct buffer_t {
  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(buffer_t);

  ///////////////////////////////////////////////////////////////////////
  buffer_t(sys_mem_heap_t& heap, size_t buffer_size, size_t heap_offset = 0)
      : ptr_(heap.create_buffer(buffer_size, heap_offset)), size_(buffer_size), heap_(&heap) {}

  ///////////////////////////////////////////////////////////////////////
  buffer_t(com_ptr<ID3D12Device3> device, size_t buffer_size)
      : owned_heap_(std::make_unique<sys_mem_heap_t>(device, buffer_size)), ptr_(owned_heap_->create_buffer(buffer_size, 0)), size_(buffer_size), heap_(owned_heap_.get()) {}

  ///////////////////////////////////////////////////////////////////////
  std::unique_ptr<void, void (*)(HANDLE)> create_shared_handle() const {
    auto device = heap_->device();

    HANDLE shared_handle = nullptr;
    if (FAILED(device->CreateSharedHandle(ptr_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle))) {
      OVEP_TESTS_THROW("Failed to create shared handle from Heap");
    }
    return std::unique_ptr<void, void (*)(HANDLE)>(
        shared_handle,
        [](HANDLE h) {
          if (h) {
            CloseHandle(h);
          }
        });
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T>
  void fill(const std::vector<T>& data, size_t offset_bytes = 0) {
    heap_->fill(data, offset_bytes);
  }

  ///////////////////////////////////////////////////////////////////////
  template <typename T>
  std::vector<T> get(size_t count, size_t offset_bytes = 0) const {
    return heap_->template get<T>(count, offset_bytes);
  }

 private:
  std::unique_ptr<sys_mem_heap_t> owned_heap_;
  com_ptr<ID3D12Resource> ptr_;
  size_t size_ = 0;
  sys_mem_heap_t* heap_ = nullptr;
};

///////////////////////////////////////////////////////////////////////
inline com_ptr<ID3D12Device3> create_device() {
  com_ptr<ID3D12Device3> device;

  if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
    return device;
  }

  OVEP_TESTS_THROW("Failed to create D3D12 Device");
}

}  // namespace d3d12_utils
}  // namespace ovep_tests

#endif
