// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <shared_mutex>
#include <span>

#include "openvino/runtime/core.hpp"
#include "ov_bin_format.h"
#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

// Forward declaration
class SharedContext;

// Type-erased owning byte buffer. `data` views the bytes; `owner` keeps them alive.
//
// This is what callers hand to BinManager / SharedContext on the deserialize path. It works
// uniformly whether the bytes came from a memory-mapped file, an ONNX attribute string, or
// memory allocated through an EP-supplied (possibly device-aware) OrtAllocator. The owner's
// deleter does the right thing in each case; the bin manager keeps it alive for as long as
// any ov::Tensor view it produced is reachable.
struct OwnedBuffer {
  std::span<const uint8_t> data;
  std::shared_ptr<void> owner;
};

// Manages native compiled model blobs and binary file serialization/deserialization
class BinManager {
 public:
  BinManager() = default;
  BinManager(const std::filesystem::path& external_bin_path) : external_bin_path_(external_bin_path) {}
  ~BinManager() = default;

  // Blob management
  void AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model);
  const ov::Tensor GetNativeBlob(const std::string& blob_name) const;

  // Model-relative path to the external weights file the blobs were compiled against. Set only for
  // weightless exports, so an empty path means the blobs are not weightless
  void SetWeightsFile(const std::filesystem::path& weights_file);
  std::filesystem::path GetWeightsFile() const;

  // Serialization
  void Serialize(std::ostream& stream, std::shared_ptr<SharedContext> shared_context = nullptr);
  void Serialize(std::shared_ptr<SharedContext> shared_context = nullptr);
  std::string SerializeToString(std::shared_ptr<SharedContext> shared_context = nullptr);

  // Deserialization: accepts any externally-owned byte buffer and views into it without copying.
  void Deserialize(OwnedBuffer buffer, std::shared_ptr<SharedContext> shared_context = nullptr);
  // Reads the external bin file via ov::read_tensor_data and feeds it through the OwnedBuffer path.
  void Deserialize(std::shared_ptr<SharedContext> shared_context = nullptr);

  // Path management
  std::filesystem::path GetExternalBinPath() const {
    std::shared_lock lock(mutex_);
    return external_bin_path_.value_or("");
  }

  // Compatibility is gated on the major component only — a major bump is a breaking change to the
  // on-disk format. The minor component is recorded in the compat string for diagnostics but does
  // not affect compatibility (additive changes remain readable by current and future builds).
  static bool IsBinVersionSupported(const std::string& model_bin_compat_ver) {
    auto parse_major = [](const std::string& ver) -> int {
      try {
        return std::stoi(ver.substr(0, ver.find('.')));
      } catch (...) {
        return -1;
      }
    };

    int model_major = parse_major(model_bin_compat_ver);
    int current_major = parse_major(bin_format::GetBinCompatVersion());

    if (model_major < 0 || current_major < 0) return false;
    return model_major == current_major;
  }

  static std::filesystem::path GetBinPathForModel(const std::filesystem::path& model_path) {
    auto result = model_path.parent_path() / model_path.stem();
    result += ("_" + std::string(kOpenVINOExecutionProvider) + ".bin");
    return result;
  }

 private:
  struct BlobContainer {
    ov::CompiledModel compiled_model;
    ov::Tensor tensor;
  };

  void DeserializeImpl(std::span<const uint8_t> data, const std::shared_ptr<SharedContext>& shared_context = nullptr);

  mutable std::shared_mutex mutex_;
  std::optional<std::filesystem::path> external_bin_path_;
  std::shared_ptr<void> buffer_owner_;  // Keeps the bytes underlying mapped_bin_/blob tensors alive.
  ov::Tensor mapped_bin_;               // Non-owning view over buffer_owner_; ROI tensors slice into this.
  std::unordered_map<std::string, BlobContainer> native_blobs_;
  std::filesystem::path weights_file_;
  bool is_deserialized_ = false;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
