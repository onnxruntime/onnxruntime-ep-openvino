// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <mutex>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_utils.h"
#include "ov_bin_manager.h"
#include "weak_singleton.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class WeightFileManager;

class SharedContext : public std::enable_shared_from_this<SharedContext> {
 public:
  explicit SharedContext(const std::filesystem::path& bin_path);
  SharedContext() : SharedContext("") {}

  struct Metadata {
    struct Value {
      struct {
        std::filesystem::path location{};
        size_t data_offset{0};
        size_t size{0};
      } serialized;

      std::unordered_map<std::string, std::shared_ptr<const ov::Tensor>> device_tensors;
    };
    using Map = std::unordered_map<std::string, Value>;
  };

  bool IsSharedWeight(const std::string& name) const {
    std::shared_lock lock(mutex_);
    return metadata_.contains(name);
  }

  void AddExternalWeight(const std::string& name, size_t offset, size_t size, const std::filesystem::path& location) {
    std::unique_lock lock(mutex_);
    auto it = metadata_.find(name);
    if (it != metadata_.end()) {
      const auto& existing = it->second.serialized;
      OVEP_ENFORCE(existing.data_offset == offset && existing.size == size && existing.location == location,
                   "AddExternalWeight: Mismatch for existing weight '", name,
                   "'. Expected offset=", existing.data_offset, " size=", existing.size,
                   " location=", ToUtf8String(existing.location),
                   ", got offset=", offset, " size=", size, " location=", ToUtf8String(location));
      return;
    }

    Metadata::Value value;
    value.serialized.data_offset = offset;
    value.serialized.size = size;
    value.serialized.location = location;
    metadata_.emplace(name, std::move(value));
  }

  Metadata::Map GetMetadataCopy() const {
    std::shared_lock lock(mutex_);
    return metadata_;
  }

  void SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const std::filesystem::path& model_dir,
                                      const Ort::Logger& logger);

  void AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model) {
    bin_manager_.AddNativeBlob(name, compiled_model);
  }

  const ov::Tensor GetNativeBlob(const std::string& blob_name) const {
    return bin_manager_.GetNativeBlob(blob_name);
  }

  void SetWeightsFile(const std::filesystem::path& weights_file) {
    bin_manager_.SetWeightsFile(weights_file);
  }

  std::filesystem::path GetWeightsFile() const {
    return bin_manager_.GetWeightsFile();
  }

  void Serialize(std::ostream& stream);
  void Serialize();
  void Deserialize(OwnedBuffer buffer);
  void Deserialize();
  std::string SerializeToString();

  std::filesystem::path GetBinPath() const {
    return bin_manager_.GetExternalBinPath();
  }

  struct WeightsFile : public std::enable_shared_from_this<WeightsFile> {
    OVEP_DISABLE_COPY_AND_MOVE(WeightsFile);
    WeightsFile() = delete;
    virtual ~WeightsFile() = default;
    explicit WeightsFile(const std::filesystem::path& filename);
    ov::Tensor GetOrCreateCpuMappedTensor() const;
    // Returns a zero-copy device-mapped tensor viewing one weight, or nullopt when the weight cannot
    // be mapped (dGPU, or a CPU_VA weight that is not cacheline-aligned) so the caller copies it.
    std::optional<ov::Tensor> TryCreateDeviceTensor(std::optional<ov::RemoteContext>& remote_context,
                                                    std::optional<ov::device::Type> device_type,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& dimensions, size_t data_offset, size_t size,
                                                    const Ort::Logger& logger);
    size_t Size() const { return file_size_; }
    size_t SizeAligned() const { return AlignUp(file_size_, 4 * 1024); }

   private:
    const void* TryGetOrCreateDeviceMapping(std::optional<ov::RemoteContext>& remote_context,
                                            std::optional<ov::device::Type> device_type,
                                            const Ort::Logger& logger);
    const std::filesystem::path file_path_;
    mutable std::optional<ov::Tensor> mapped_tensor_;  // Lazily memory-mapped tensor of the entire file
    mutable std::mutex mapped_tensor_mutex_;
    const size_t file_size_{0};
    struct MappingContainer {
      const void* ptr_{nullptr};
      ov::Tensor tensor_;
    };
    mutable std::mutex mutex_;
    std::map<std::string, MappingContainer> imported_device_tensors_;
  };

 private:
  void LoadTensorFromFile(
      Metadata::Value& value,
      const std::filesystem::path& model_dir,
      std::optional<ov::RemoteContext>& remote_context,
      std::optional<ov::device::Type> device_type,
      const ov::element::Type& element_type,
      const ov::Shape& dimensions,
      const Ort::Logger& logger);

  mutable std::shared_mutex mutex_;
  std::filesystem::path bin_path_;
  BinManager bin_manager_;
  std::shared_ptr<WeightFileManager> weight_file_manager_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<WeightsFile>> weight_files_;
  Metadata::Map metadata_;
};

class WeightFileManager : public WeakSingleton<WeightFileManager> {
 public:
  using WeightsFile = SharedContext::WeightsFile;
  std::shared_ptr<WeightsFile> GetOrCreateWeightsFile(const std::filesystem::path& weights_path) {
    auto absolute_path = std::filesystem::absolute(weights_path);
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = files_.try_emplace(absolute_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<WeightsFile>(absolute_path);
    }
    return it->second;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<WeightsFile>> files_;
};

/**
 * @class SharedContextManager
 * @brief Thread-safe manager for OpenVINO EP shared contexts containing compiled models and weights.
 *
 * Supports two modes:
 * - **Active Context**: Tracked context for EP models with sharing enabled (ep_ctx.share_).
 *   Use GetOrCreateActiveSharedContext() and clear with ClearActiveSharedContext().
 * - **Passive Context**: Standard contexts that persist until no  longer referenced.
 *   Use GetOrCreateSharedContext() for regular usage or loading existing EP contexts.
 */
class SharedContextManager {
 public:
  std::shared_ptr<SharedContext> GetOrCreateActiveSharedContext(const std::filesystem::path& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_context_) {
      return active_context_;
    }
    auto [it, inserted] = contexts_.try_emplace(model_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<SharedContext>(model_path);
    }
    active_context_ = it->second;
    active_context_path_ = model_path;
    return it->second;
  }

  std::shared_ptr<SharedContext> GetOrCreateSharedContext(const std::filesystem::path& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = contexts_.try_emplace(model_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<SharedContext>(model_path);
    }
    return it->second;
  }

  void ClearActiveSharedContext() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_context_) {
      contexts_.erase(active_context_path_);
      active_context_path_.clear();
    }
    active_context_ = nullptr;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<SharedContext>> contexts_;
  std::shared_ptr<SharedContext> active_context_;
  std::filesystem::path active_context_path_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
