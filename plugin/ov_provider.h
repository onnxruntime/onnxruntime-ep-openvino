// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <vector>
#include <string>
#include <string_view>
#include <span>
#include <functional>
#include <set>
#include <unordered_set>
#include <filesystem>

#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ov_plugin_utils.h"
#include "ov_plugin_ver.h"

#include "openvino/runtime/core.hpp"
#include "openvino/frontend/manager.hpp"
#include "ov_custom_allocator.h"
#include "ov_shared_context.h"
#include "ov_bin_manager.h"
#include "ov_device.h"
#include "ov_plugin_options.h"

struct OrtEpProfilerImpl;

namespace onnxruntime {
namespace openvino_ep_plugin {

struct ModelTransformation {
  using InferRequestInitializer = typename std::function<void(ov::InferRequest&)>;
  using TransformFunc = typename std::function<OrtStatus*(std::shared_ptr<ov::Model>&)>;

  std::string name_;
  TransformFunc transform_func_;
  InferRequestInitializer infer_request_initializer_;  // optional will be checked before use

  ModelTransformation(std::string_view name, TransformFunc func, InferRequestInitializer initializer = nullptr) : name_(name), transform_func_(std::move(func)), infer_request_initializer_(std::move(initializer)) {}
};

struct TransformationPipelineOptions {
  bool dump_before_transformations{false};
  bool dump_after_transformations{false};
  std::string dump_prefix;
  std::filesystem::path dump_dir;
};

class TransformationPipeline
    : public std::vector<ModelTransformation> {
 public:
  using Base = std::vector<ModelTransformation>;
  using Base::Base;

  TransformationPipeline(const TransformationPipelineOptions& options = {}) : options_(options) {}

  OrtStatus* ApplyTransformations(std::shared_ptr<ov::Model>& model) const;
  void InitializeInferRequest(ov::InferRequest& infer_request) const;
  TransformationPipelineOptions options_;
};

struct OvComputeInfo;
struct OnnxIOMapping;
struct EpContextNode;
class OpenVINOEpPlugin : public OrtEp {
 public:
  struct MemoryInfos {
    Ort::ConstMemoryInfo default_{nullptr};
    Ort::ConstMemoryInfo cpu_input_{nullptr};
    Ort::ConstMemoryInfo cpu_output_{nullptr};
  };

  OpenVINOEpPlugin(const std::string& name,
                   const OpenVINOEpPluginOptions& options,
                   const OrtLogger& logger,
                   OvDevice ov_device,
                   std::shared_ptr<ov::Core> ov_core,
                   std::shared_ptr<ov::frontend::FrontEndManager> onnx_fe_manager,
                   MemoryInfos memory_infos);
  ~OpenVINOEpPlugin();

  // Version-agnostic forms of the ORT v28+ EPContext named-buffer callbacks.
  using EpContextWriteFunc = std::function<void(std::string_view name, std::span<const uint8_t> data)>;
  using EpContextReadFunc = std::function<OwnedBuffer(std::string_view name)>;

  // Wraps the raw ORT v28+ callback pointers/state into EpContextWriteFunc / EpContextReadFunc.
  // Templated so this header never has to name OrtWriteNamedBufferFunc / OrtReadNamedBufferFunc.
  // The read wrapper also fetches/creates the EP's EPContext allocator; both wrappers throw on
  // host callback failure instead of returning a status.
  template <typename WriteFn, typename ReadFn>
  void SetRawEpContextDataCallbacks(WriteFn write_fn, void* write_state, ReadFn read_fn, void* read_state) {
    EpContextWriteFunc wrapped_write;
    if (write_fn != nullptr) {
      wrapped_write = [write_fn, write_state](std::string_view name, std::span<const uint8_t> data) {
        const std::string name_str(name);
        Ort::ThrowOnError(write_fn(write_state, name_str.c_str(), data.data(), data.size()));
      };
    }

    EpContextReadFunc wrapped_read;
    if (read_fn != nullptr) {
      wrapped_read = [this, read_fn, read_state](std::string_view name) -> OwnedBuffer {
        const std::string name_str(name);
        auto ep_ctx_alloc = GetOrCreateEpCtxAllocator();
        OrtAllocator* allocator = ep_ctx_alloc ? ep_ctx_alloc.get() : nullptr;
        if (allocator == nullptr) {
          Ort::ThrowOnError(ort_api.GetAllocatorWithDefaultOptions(&allocator));
        }
        void* raw_buf = nullptr;
        size_t buf_size = 0;
        Ort::ThrowOnError(read_fn(read_state, name_str.c_str(), allocator, &raw_buf, &buf_size));
        // ep_ctx_alloc is captured so the OVRTAllocator (if any) outlives this buffer even if
        // the provider is torn down before the SharedContext releases it.
        auto owner = std::shared_ptr<void>(raw_buf, [ep_ctx_alloc, allocator](void* p) { allocator->Free(allocator, p); });
        return OwnedBuffer{
            .data = std::span<const uint8_t>(static_cast<const uint8_t*>(raw_buf), buf_size),
            .owner = std::move(owner),
        };
      };
    }

    ep_ctx_write_fn_ = std::move(wrapped_write);
    ep_ctx_read_fn_ = std::move(wrapped_read);
  }

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPlugin)

  struct ModelInitializerInfo;
  struct ModelFilePaths {
    std::filesystem::path model_path;
    std::filesystem::path output_model_path;
  };

  const char* GetName() const noexcept {
    return name_.c_str();
  }

  Ort::Logger GetLogger() const noexcept {
    return logger_;
  }

  OrtStatus* GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info);
  OrtStatus* GetDefaultMemoryDevice(const OrtMemoryDevice** device) const;
  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info, OrtAllocator** allocator);
  OrtStatus* CreateProfiler(OrtEpProfilerImpl** profiler);
  OrtStatus* GetWeightlessSupport(OrtWeightlessSupport* support) const;
  OrtStatus* Compile(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes);
  OrtStatus* CompileEpContextNode(const OrtGraph& graph, const OrtNode& fused_node, OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node);
  void ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos, size_t num_node_compute_infos);

  const char* GetCompiledModelCompatibilityInfo(const OrtGraph* graph);

  OrtStatus* InitializeSharedContext(const OrtGraph* graph);

  TransformationPipeline BuildTransformationPipeline(const ModelFilePaths& paths, const std::string& output_node_name);
  OrtStatus* SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options);

  OrtStatus* GetFilePaths(const OrtGraph* graph, ModelFilePaths& paths) const;

  static constexpr const char* provider_name_ = kOpenVINOExecutionProvider;

 private:
  // Lazily creates and returns the OVRTAllocator for EPContext reads.
  // Returns null when memory_infos_.cpu_input_ is null (meta-device / virtual factory).
  std::shared_ptr<OVRTAllocator> GetOrCreateEpCtxAllocator() const;

 private:
  std::string name_;
  const OrtApi& ort_api{Ort::GetApi()};
  const OrtEpApi& ep_api{Ort::GetEpApi()};
  Ort::Logger logger_;
  const OvDevice ov_device_;
  std::shared_ptr<ov::Core> ov_core_;

  mutable std::mutex ep_ctx_allocator_mutex_;
  mutable std::shared_ptr<OVRTAllocator> ep_ctx_allocator_;

  std::shared_ptr<SharedContextManager> shared_context_manager_;
  std::shared_ptr<SharedContext> shared_context_;
  std::shared_ptr<ov::frontend::FrontEndManager> onnx_fe_manager_;

  const OpenVINOEpPluginOptions options_;
  std::atomic<bool> ov_profiling_enabled_{false};
  std::set<OvComputeInfo*> computes_;
  std::string cached_compatibility_info_;
  MemoryInfos memory_infos_;

  EpContextWriteFunc ep_ctx_write_fn_;
  EpContextReadFunc ep_ctx_read_fn_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
