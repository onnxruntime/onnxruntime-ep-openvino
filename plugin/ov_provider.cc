// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <atomic>
#include <format>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "ov_provider.h"
#include "ov_bin_format.h"
#include "ov_compute.h"
#include "ov_ep_profiler.h"
#include "ov_compute_stateful.h"
#include "ov_ep_context.h"
#include "weak_singleton.h"
#include "ov_supported_ops.h"
#include "ov_plugin_utils.h"
#include "onnxruntime/onnxruntime_session_options_config_keys.h"
#include "transformations/ov_weights_as_input.h"
#include "ov_custom_allocator.h"
#include "ov_version.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/runtime/properties.hpp"

using namespace onnxruntime::openvino_ep_plugin;

using shared_context_manager_singleton = WeakSingleton<SharedContextManager>;

static std::filesystem::path ResolveDumpDirectory(const std::filesystem::path& configured_output_path,
                                                  bool treat_non_existing_path_as_file) {
  // Prefer filesystem metadata over path shape so dotted directory names work.
  std::error_code ec;
  const auto file_status = std::filesystem::status(configured_output_path, ec);
  if (!ec) {
    if (std::filesystem::is_directory(file_status)) {
      return configured_output_path;
    }
    if (std::filesystem::is_regular_file(file_status)) {
      return configured_output_path.parent_path();
    }
  }

  return treat_non_existing_path_as_file ? configured_output_path.parent_path() : configured_output_path;
}

static OrtStatus* EnsureDumpDirectoryExists(const std::filesystem::path& dump_dir) {
  if (dump_dir.empty()) {
    return OVEP_MAKE_STATUS("Dump directory path is empty.", ORT_EP_FAIL);
  }

  std::error_code ec;
  if (std::filesystem::exists(dump_dir, ec)) {
    if (ec) {
      return OVEP_MAKE_STATUS(std::format("Failed to query dump directory '{}': {}", ToUtf8String(dump_dir), ec.message()).c_str(), ORT_EP_FAIL);
    }
    if (!std::filesystem::is_directory(dump_dir, ec) || ec) {
      return OVEP_MAKE_STATUS(std::format("Dump path '{}' exists but is not a directory.", ToUtf8String(dump_dir)).c_str(), ORT_EP_FAIL);
    }
    return nullptr;
  }
  if (ec) {
    return OVEP_MAKE_STATUS(std::format("Failed to query dump directory '{}': {}", ToUtf8String(dump_dir), ec.message()).c_str(), ORT_EP_FAIL);
  }

  std::filesystem::create_directories(dump_dir, ec);
  if (ec || !std::filesystem::is_directory(dump_dir, ec)) {
    return OVEP_MAKE_STATUS(std::format("Failed to create dump directory '{}': {}", ToUtf8String(dump_dir), ec.message()).c_str(), ORT_EP_FAIL);
  }

  return nullptr;
}

static std::filesystem::path GetDumpDirectory(const OpenVINOEpPluginOptions& options,
                                              const std::filesystem::path& fallback_output_path) {
  if (!options.provider_options_.dump_subgraphs_path_.empty()) {
    return ResolveDumpDirectory(options.provider_options_.dump_subgraphs_path_, false);
  }
  // The fallback path originates from the model output path, which is usually a file path.
  return ResolveDumpDirectory(fallback_output_path, true);
}

static bool IsDumpSubgraphsEnabled(const OpenVINOEpPluginOptions& options) {
  return !options.provider_options_.dump_subgraphs_path_.empty();
}

OrtStatus* TransformationPipeline::ApplyTransformations(std::shared_ptr<ov::Model>& model) const {
  if (options_.dump_before_transformations || options_.dump_after_transformations) {
    OVEP_RETURN_IF_ERROR(EnsureDumpDirectoryExists(options_.dump_dir));
  }

  for (const auto& transformation : *this) {
    if (options_.dump_before_transformations) {
      auto&& dump_path = options_.dump_dir / (options_.dump_prefix + "_before_transformation_" + transformation.name_ + ".xml");
      ov::serialize(model, dump_path);
    }

    OVEP_ENFORCE(transformation.transform_func_,
                 "Transform function is not set for the model transformation: ", transformation.name_);
    OVEP_RETURN_IF_ERROR(transformation.transform_func_(model));

    if (options_.dump_after_transformations) {
      auto&& dump_path = options_.dump_dir / (options_.dump_prefix + "_after_transformation_" + transformation.name_ + ".xml");
      ov::serialize(model, dump_path);
    }
  }
  return nullptr;
}

void TransformationPipeline::InitializeInferRequest(ov::InferRequest& infer_request) const {
  for (const auto& transformation : *this) {
    if (transformation.infer_request_initializer_) {
      transformation.infer_request_initializer_(infer_request);
    }
  }
}

// Implementation class definition
OpenVINOEpPlugin::OpenVINOEpPlugin(const std::string& name,
                                   const OpenVINOEpPluginOptions& options,
                                   const OrtLogger& logger,
                                   OvDevice ov_device,
                                   std::shared_ptr<ov::Core> ov_core,
                                   std::shared_ptr<ov::frontend::FrontEndManager> onnx_fe_manager,
                                   MemoryInfos memory_infos)
    : OrtEp{},
      name_(name),
      logger_(&logger),
      ov_device_(std::move(ov_device)),
      ov_core_(std::move(ov_core)),
      shared_context_manager_(shared_context_manager_singleton::Get()),
      onnx_fe_manager_(std::move(onnx_fe_manager)),
      options_(options),
      memory_infos_(std::move(memory_infos)) {
  ort_version_supported = ort_runtime_api_version;

  OrtEp::GetName = OVEP_API_ENTRY(OpenVINOEpPlugin, GetName);
  OrtEp::GetCapability = OVEP_API_ENTRY(OpenVINOEpPlugin, GetCapability);
  OrtEp::CreateAllocator = OVEP_API_ENTRY(OpenVINOEpPlugin, CreateAllocator);
  OrtEp::Compile = OVEP_API_ENTRY_EX(OpenVINOEpPlugin, Compile, ORT_EP_FAIL);
  OrtEp::ReleaseNodeComputeInfos = OVEP_API_ENTRY_TEARDOWN(OpenVINOEpPlugin, ReleaseNodeComputeInfos);
  OrtEp::SetDynamicOptions = OVEP_API_ENTRY(OpenVINOEpPlugin, SetDynamicOptions);
  OrtEp::GetCompiledModelCompatibilityInfo = OVEP_API_ENTRY(OpenVINOEpPlugin, GetCompiledModelCompatibilityInfo);

  // Opt in to the v27 default-device callback whenever we have a host-accessible (or default)
  // memory info registered. The default device now resolves to host-accessible memory.
  // Meta-device factories leave all memory infos null and intentionally fall through to ORT's
  // legacy CPU default.
  if (!features::disable_device_specific_allocators.is_enabled()) {
    OrtVersionGate<27>(static_cast<OrtEp*>(this), [this](auto* ep) {
      if (memory_infos_.default_ != nullptr || memory_infos_.cpu_output_ != nullptr) {
        ep->GetDefaultMemoryDevice = OVEP_API_ENTRY(OpenVINOEpPlugin, GetDefaultMemoryDevice);
      }
      ep->CreateAllocator = nullptr;  // For 1.27+ we use the factory-level CreateAllocator
    });
  }
  OrtVersionGate<25>(static_cast<OrtEp*>(this), [this](auto* ep) {
    ep->CreateProfiler = OVEP_API_ENTRY(OpenVINOEpPlugin, CreateProfiler);
  });
  OrtVersionGate<29>(static_cast<OrtEp*>(this), [this](auto* ep) {
    ep->GetWeightlessSupport = OVEP_API_ENTRY(OpenVINOEpPlugin, GetWeightlessSupport);
  });
}

OrtStatus* OpenVINOEpPlugin::GetDefaultMemoryDevice(const OrtMemoryDevice** device) const {
  // The default device resolves to host-accessible memory when no device-typed DEFAULT info is
  // registered (the common case now). cpu_input_ and cpu_output_ point at the same HOST_ACCESSIBLE info.
  Ort::ConstMemoryInfo mem_info =
      memory_infos_.default_ != nullptr ? memory_infos_.default_ : memory_infos_.cpu_output_;
  *device = ep_api.MemoryInfo_GetMemoryDevice(mem_info);
  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::CreateAllocator(const OrtMemoryInfo* memory_info, OrtAllocator** allocator) {
  OVEP_ENFORCE(features::disable_device_specific_allocators.is_enabled() || ort_runtime_api_version < 27,
               "CreateAllocator should not be called when running with ORT API v27 or above");
  *allocator = new OVRTAllocator(*ov_core_, *memory_info, ov_device_, logger_);
  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::GetWeightlessSupport(OrtWeightlessSupport* support) const {
  // Weightless + shared context is not supported yet. Only NPU supports weightless blobs.
  if (options_.ep_ctx_.share_ || !DeviceSupportsWeightless(ov_device_.c_str())) {
    *support = OrtWeightlessSupport_NONE;
    return nullptr;
  }

  // External-only weightless- fully weightless pending stable Constant IDs.
  *support = OrtWeightlessSupport_EXTERNAL_ONLY;
  return nullptr;
}

OpenVINOEpPlugin::~OpenVINOEpPlugin() = default;

std::shared_ptr<OVRTAllocator> OpenVINOEpPlugin::GetOrCreateEpCtxAllocator() const {
  if (memory_infos_.cpu_input_ == nullptr) {
    return nullptr;
  }
  std::lock_guard lock(ep_ctx_allocator_mutex_);
  if (!ep_ctx_allocator_) {
    ep_ctx_allocator_ = std::make_shared<OVRTAllocator>(*ov_core_, *memory_infos_.cpu_input_, ov_device_, logger_);
  }
  return ep_ctx_allocator_;
}

OrtStatus* OpenVINOEpPlugin::GetFilePaths(const OrtGraph* graph, ModelFilePaths& paths) const {
  const ORTCHAR_T* char_model_path = nullptr;
  OVEP_RETURN_IF_ERROR(ort_api.Graph_GetModelPath(graph, &char_model_path));
  paths.model_path = char_model_path;
  if (paths.model_path.empty()) {
    // We infer that this is running an ep context model from memory without external weights.
    // In this case we find the model path is ep_context path
    paths.model_path = options_.ep_ctx_.path_;
  }
  // Output path is the EP context path if specified -- otherwise the same as model path.
  paths.output_model_path = options_.ep_ctx_.path_;
  if (paths.output_model_path.empty()) {
    paths.output_model_path = paths.model_path;
  }

  return nullptr;
}

static bool IsOvEpContextNode(const EpContextNode& ep_context) {
  return ep_context.source == OpenVINOEpPlugin::provider_name_;
}

const char* OpenVINOEpPlugin::GetCompiledModelCompatibilityInfo(const OrtGraph* /*graph*/) {
  cached_compatibility_info_.clear();

  std::string ov_compat_string;
  if (!computes_.empty()) {
    auto* first_compute = *computes_.begin();
    try {
      ov_compat_string = first_compute->GetCompiledModel().get_property(ov::runtime_requirements);
    } catch (const ov::Exception& e) {
      ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "ov::runtime_requirements not supported: %s", e.what());
    }
  }

  nlohmann::ordered_json j;
  j["ovep_compat_ver"] = "1.0";
  j["ovep_ver"] = OVEP_PLUGIN_VER_TAGGED;
  j["ovep_bin_ver"] = bin_format::GetBinCompatVersion();
  j["ov_compat_string"] = std::move(ov_compat_string);
  cached_compatibility_info_ = j.dump();

  ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_INFO, "compatibility_info='%s'", cached_compatibility_info_.c_str());
  return cached_compatibility_info_.c_str();
}

OrtStatus* OpenVINOEpPlugin::InitializeSharedContext(const OrtGraph* ort_graph) {
  OVEP_TRACE(logger_);

  ModelFilePaths paths;
  OVEP_RETURN_IF_ERROR(GetFilePaths(ort_graph, paths));
  bool expected_to_be_initalized = false;

  // Create any shared contexts from native ep context nodes.
  Ort::ConstGraph graph{ort_graph};
  for (const auto& node : graph.GetNodes()) {
    bool is_ep_context = EpContextNode::IsEpContextNode(node);
    if (!is_ep_context) {
      continue;
    }

    EpContextNode ep_context(node, paths.output_model_path, paths.model_path);
    if (IsOvEpContextNode(ep_context) &&
        ep_context.private_fields_.type == EpContextNode::EpContextType::Native) {
      expected_to_be_initalized = true;
      if (ep_context.embed_mode) {
        if (ep_context.main_context) {
          OVEP_RETURN_STATUS_IF(shared_context_ != nullptr, ORT_INVALID_GRAPH,
                                "Multiple embedded native EPContext main contexts are not supported.");
          {
            // Each EP instance needs its own SharedContext in embed mode.
            // The blob data is stored inline in BinManager::embedded_string_;
            // sharing one context across sessions causes an access violation
            // when a later Deserialize() overwrites the buffer that earlier
            // sessions' compiled models still reference.
            // Since we don't allow embed_mode with ep context sharing, it's safe
            // to create unique shared context for each ep context node.
            OVEP_TRACE(logger_, "Creating new SharedContext for embed mode");
            shared_context_ = std::make_shared<SharedContext>();
          }
          {
            OVEP_TRACE(logger_, "SharedContext::Deserialize(embed_data)");
            // Move the embed-mode bytes into a shared_ptr<string> owner; the OwnedBuffer
            // view points into that owner so no extra copy lands in BinManager.
            auto owner = std::make_shared<std::string>(std::move(ep_context.ep_cache_context));
            OwnedBuffer buffer{
                .data = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(owner->data()), owner->size()),
                .owner = std::shared_ptr<void>(owner, owner.get()),
            };
            shared_context_->Deserialize(std::move(buffer));
            break;  // Embed-mode contexts should not have additional blobs, so we can stop after the first one.
          }
        }
      } else {
        const auto bin_path = ep_context.GetEpContextPath();
        if (ep_ctx_read_fn_) {
          // `bin_path` is a host-chosen key, not a file on disk, and distinct models can collide
          // on it. Use a private SharedContext so the path-keyed cache can't hand this session a
          // different model's already-deserialized context; the read callback self-delivers our
          // full blob anyway.
          OVEP_TRACE(logger_, "SharedContext::Deserialize(via EPContext read callback)");
          shared_context_ = std::make_shared<SharedContext>(bin_path);
          const std::string file_name = ToUtf8String(bin_path.filename());
          // Throws on host callback failure; hand the allocator-owned memory directly to
          // SharedContext so the bin manager can build a non-owning ov::Tensor view over these
          // bytes without an extra copy.
          OwnedBuffer buffer = ep_ctx_read_fn_(file_name);
          OVEP_RETURN_IF(buffer.data.empty(), ort_api,
                         std::format("EPContext read callback returned an empty buffer for '{}'", file_name).c_str());
          shared_context_->Deserialize(std::move(buffer));
        } else {
          OVEP_TRACE(logger_, "SharedContextManager::GetOrCreateSharedContext(path=%s)", ToUtf8String(bin_path).c_str());
          shared_context_ = shared_context_manager_->GetOrCreateSharedContext(bin_path);
          OVEP_TRACE(logger_, "SharedContext::Deserialize()");
          shared_context_->Deserialize();
        }
        break;  // Stop after the first native EP context node since they should all point to the same shared context.
      }
    }
  }

  OVEP_ENFORCE(!expected_to_be_initalized || shared_context_, "Did not find binary information in ep context nodes.");

  // No ep context nodes found - create a shared context that can hold native blobs or shared weights.
  if (!shared_context_) {
    const auto bin_path = BinManager::GetBinPathForModel(paths.output_model_path);
    if (options_.ep_ctx_.enable_ && options_.ep_ctx_.share_) {
      // We're creating a shared ep context model set: get or create the active context.
      OVEP_TRACE(logger_, "SharedContextManager::GetOrCreateActiveSharedContext(bin_path=%s)", ToUtf8String(bin_path).c_str());
      shared_context_ = shared_context_manager_->GetOrCreateActiveSharedContext(bin_path);
    } else if (options_.ep_ctx_.share_) {
      OVEP_TRACE(logger_, "SharedContextManager::GetOrCreateSharedContext(bin_path=%s)", ToUtf8String(bin_path).c_str());
      shared_context_ = shared_context_manager_->GetOrCreateSharedContext(bin_path);
    } else {
      // Keep generated blobs out of the path cache used to deserialize EPContext models.
      OVEP_TRACE(logger_, "Creating new SharedContext(bin_path=%s)", ToUtf8String(bin_path).c_str());
      shared_context_ = std::make_shared<SharedContext>(bin_path);
    }
  }

  return nullptr;
}

// Checks that a node and all nodes within its subgraphs (If/Loop/Scan) are supported.
static bool NodeAndSubnodesSupported(const Ort::ConstNode& node, const OpenVINOEpProviderOptions& options) {
  // The hint only adds support; it must not veto GQA once the supported-op list enables it.
  const bool gqa_opt_in = options.hint_enable_gqa_ &&
                          node.GetOperatorType() == std::string_view{"GroupQueryAttention"} &&
                          node.GetDomain() == std::string_view{"com.microsoft"} &&
                          OpenVINOVersion::GetBuiltVersion() >= OpenVINOVersion(2026, 3);
  if (!SupportedOps::Get().IsOpSupported(node) && !gqa_opt_in) {
    return false;
  }

  for (const auto& [attr_name, ort_subgraph] : node.GetSubgraphs()) {
    for (const auto& sub_node : Ort::ConstGraph{ort_subgraph}.GetNodes()) {
      if (!NodeAndSubnodesSupported(sub_node, options)) {
        return false;
      }
    }
  }
  return true;
}

OrtStatus* OpenVINOEpPlugin::GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  Ort::ConstGraph ort_graph{graph};

  // ORT calls GetCapability on nested subgraphs (If/Loop/Scan branches) separately.
  // Returning nullptr lets OV compile the parent graph with subgraph attributes intact.
  if (ort_graph.GetParentNode() != nullptr) {
    return nullptr;
  }

  std::vector<Ort::ConstNode> nodes = ort_graph.GetNodes();

  ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_INFO, "GetCapability total_nodes=%zu", nodes.size());

  ModelFilePaths paths;
  OVEP_RETURN_IF_ERROR(GetFilePaths(graph, paths));

  auto add_nodes_to_fuse = [this, graph_support_info](const std::vector<const OrtNode*>& nodes) -> OrtStatus* {
    OrtNodeFusionOptions options{};
    options.ort_version_supported = ort_runtime_api_version;
    options.drop_constant_initializers = true;
    return ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, nodes.data(), nodes.size(), &options);
  };

  // First try to only claim ovep ep context nodes. If we find an OV EP context node, we will not claim other nodes.
  bool has_ov_ep_ctx_node = false;
  for (const auto& node : nodes) {
    if (EpContextNode::IsEpContextNode(node)) {
      EpContextNode ep_context(node, paths.output_model_path, paths.model_path);
      if (IsOvEpContextNode(ep_context)) {
        OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(std::vector<const OrtNode*>{node}));
        has_ov_ep_ctx_node = true;
      }
    }
  }

  if (!has_ov_ep_ctx_node) {
    std::vector<const OrtNode*> supported_nodes;
    for (const auto& node : nodes) {
      // We still check for ep context node since there might be ep context nodes from other providers and we want to skip those.
      if (!EpContextNode::IsEpContextNode(node) && NodeAndSubnodesSupported(node, options_.provider_options_)) {
        supported_nodes.push_back(node);
      } else {
        // Flush accumulated supported nodes before the break
        if (!supported_nodes.empty()) {
          OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
          supported_nodes.clear();
        }
      }
    }

    if (!supported_nodes.empty()) {
      OVEP_RETURN_IF_ERROR(add_nodes_to_fuse(supported_nodes));
    }
  }

  return nullptr;
}

struct OpenVINOEpPlugin::ModelInitializerInfo {
  struct InitializerInfo {
    size_t size = 0;
    int64_t offset = 0;
    std::filesystem::path external_file_path;
  };
  std::map<std::string, InitializerInfo> initializers_;

  OrtStatus* Init(Ort::ConstGraph& graph_ref) {
    const std::vector<Ort::ConstValueInfo> initializers = graph_ref.GetInitializers();

    for (const auto& initializer : initializers) {
      // Get external initializer information if available
      Ort::ExternalInitializerInfo ext_info;
      OVEP_RETURN_IF_ERROR(initializer.GetExternalInitializerInfo(ext_info));
      if (!ext_info) continue;  // not an external initializer

      ModelInitializerInfo::InitializerInfo init_info;
      init_info.external_file_path = ext_info.GetFilePath();
      if (init_info.external_file_path.empty()) {
        // For some reason we sometimes get an initializer that's claimed be external but the path is empty.
        // Skip them for now. Might need to file an ORT bug if it's not resolved on latest.
        continue;
      }
      init_info.offset = ext_info.GetFileOffset();
      init_info.size = ext_info.GetByteSize();

      initializers_[initializer.GetName()] = std::move(init_info);
    }
    return nullptr;
  }
};
using ModelInitializerInfo = OpenVINOEpPlugin::ModelInitializerInfo;

static void AddSharableWeights(const ModelInitializerInfo& model_info,
                               std::shared_ptr<SharedContext>& shared_context) {
  OVEP_ENFORCE(shared_context != nullptr, "Shared context is null when adding sharable weights.");

  for (const auto& [name, init_info] : model_info.initializers_) {
    shared_context->AddExternalWeight(
        name, static_cast<size_t>(init_info.offset),
        init_info.size,
        init_info.external_file_path);
  }
}

TransformationPipeline OpenVINOEpPlugin::BuildTransformationPipeline(const ModelFilePaths& paths, const std::string& dump_prefix) {
  OVEP_TRACE(logger_);

  TransformationPipeline transformations;
  const std::filesystem::path model_dir = paths.model_path.parent_path();

  if (options_.debug_enabled_ || IsDumpSubgraphsEnabled(options_)) {
    transformations.options_.dump_after_transformations = true;
    transformations.options_.dump_prefix = dump_prefix;
    transformations.options_.dump_dir = GetDumpDirectory(options_, paths.output_model_path);
    transformations.emplace_back("input_model",
                                 [&](std::shared_ptr<ov::Model>& /* model */) {
                                   return nullptr;  // No transformation, just want pipeline to dump model.
                                 });
  }

  if (!options_.provider_options_.reshape_input_.empty()) {
    transformations.emplace_back(
        "reshape_input",
        /* transformation function */
        [&](std::shared_ptr<ov::Model>& model) {
          model->reshape(options_.provider_options_.reshape_input_);
          return nullptr;
        });
  }

  if (options_.ep_ctx_.share_ && shared_context_) {
    transformations.emplace_back(
        "shared_weights",
        /* transformation function */
        [&](std::shared_ptr<ov::Model>& model) { return weights_as_inputs::TransformSharedWeightsToInputs(model, *shared_context_); },
        /* optional initializer */
        [&, model_dir](ov::InferRequest& ir) {
          shared_context_->SetSharedWeightsOnInferRequest(ir, model_dir, logger_);
        });
  }

  return transformations;
}

OrtStatus* OpenVINOEpPlugin::SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options) {
  // Process all options
  for (size_t i = 0; i < num_options; i++) {
    const std::string value(option_values[i]);

    if (value.empty()) {
      continue;
    }
    const std::string key(option_keys[i]);

    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
                std::format("[OVEP] SetDynamicOptions (ORT -> OVEP) | key='{}', value='{}'", key, value).c_str());

    if (key == kOrtEpDynamicOptionsWorkloadType) {
      std::string workload_type;
      if (value == "Efficient") {
        workload_type = "EFFICIENT";
      } else if (value == "Default") {
        workload_type = "DEFAULT";
      } else {
        ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "Unknown workload_type - ignoring %s / %s", key.c_str(), value.c_str());
        ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_WARNING, "Supported types are 'Efficient' and 'Default'");
      }
      if (!workload_type.empty()) {
        for (auto& ov_compute : computes_) {
          OVEP_RETURN_IF_ERROR(ov_compute->SetWorkloadType(workload_type));
        }
      }
    } else if (key == "kvcache_rewind") {
      int64_t index;
      try {
        index = std::stoll(value);
      } catch (const std::exception& e) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    std::format("Conversion for kvcache_rewind string value to int64_t index failed. Exception: {}", e.what()).c_str());
      }
      if (index < 0) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    std::format("kvcache_rewind index must be non-negative, got {}", index).c_str());
      }
      for (auto& ov_compute : computes_) {
        OVEP_RETURN_IF_ERROR(ov_compute->KVCacheRewind(static_cast<size_t>(index)));
      }
    } else if (key == "kvcache_reorder") {
      std::vector<int32_t> src_indices;
      std::vector<int32_t> dst_indices;
      OVEP_RETURN_IF_ERROR(ParseSrcAndDstIntCommaLists(ort_api, value, src_indices, dst_indices));

      for (auto& ov_compute : computes_) {
        OVEP_RETURN_IF_ERROR(ov_compute->SetReorderKVCacheStatus(src_indices, dst_indices));
      }
    }
  }

  return nullptr;
}

OrtStatus* OpenVINOEpPlugin::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes,
                                     size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) {
  // Perform up-front check to make sure that when enable_causallm_ is set, we only have one fused node to compile.
  if (options_.provider_options_.enable_causallm_ && count > 1) {
    return OVEP_MAKE_STATUS(std::format("'enable_causallm' is enabled, but the model graph has been partitioned "
                                        "into {} subgraphs. CausalLM mode requires the entire model to be "
                                        "compiled as a single subgraph.",
                                        count)
                                .c_str(),
                            ORT_EP_FAIL);
  }

  // This counter exists only to keep the blob name consistent with legacy ep.
  static std::atomic<size_t> s_op_counter = 0;
  size_t op_count = ++s_op_counter;
  bool compiled = false;

  std::function<OrtStatus*()> embed_mode_shared_context_serializer;

  for (size_t i = 0; i < count; ++i) {
    OVEP_RETURN_IF_ERROR(InitializeSharedContext(graphs[i]));
  }

  // Process all graphs
  for (size_t i = 0; i < count; ++i) {
    Ort::ConstGraph ort_graph{graphs[i]};
    Ort::ConstNode ort_fused_node{fused_nodes[i]};

    const std::vector<Ort::ConstNode> nodes = ort_graph.GetNodes();
    const size_t num_nodes = nodes.size();

    auto ov_compute = options_.provider_options_.enable_causallm_ ? std::make_unique<OvComputeInfoStateful>(*ov_core_, logger_) : std::make_unique<OvComputeInfo>(*ov_core_, logger_);

    ModelFilePaths paths;
    OVEP_RETURN_IF_ERROR(GetFilePaths(ort_graph, paths));

    const std::filesystem::path& model_path = paths.model_path;
    const std::filesystem::path& output_model_path = paths.output_model_path;

    OvModelInfo model_info{.model_path = model_path};
    OVEP_RETURN_IF_ERROR(model_info.io_mapping.Init(ort_api, *ort_graph));
    auto& io_mapping = model_info.io_mapping;

    const std::string output_node_name = std::format("{}_{}_OpenVINO-EP-subgraph_{}_{}",
                                                     ToUtf8String(output_model_path.stem()),
                                                     OpenVINOEpPlugin::provider_name_,
                                                     op_count,
                                                     i);

    auto try_export_ep_context = [&](OvComputeInfo& ov_compute, OnnxIOMapping&& io_mapping) -> OrtStatus* {
      if (!options_.ep_ctx_.enable_) {
        return nullptr;
      }

      EpContextNode node;
      node.embed_mode = options_.ep_ctx_.embed_ ? 1u : 0u;
      node.partition_name = output_node_name;

      node.private_fields_.epctx_dir = output_model_path.parent_path();
      node.private_fields_.node_name = ort_fused_node.GetName();

      node.ep_sdk_version = to_string(OpenVINOVersion::GetBuiltVersion());
      node.source = OpenVINOEpPlugin::provider_name_;
      node.onnx_model_filename = ToUtf8String(model_path.filename());

      auto&& compiled_model = ov_compute.GetCompiledModel();
      if (node.embed_mode) {
        shared_context_->AddNativeBlob(node.partition_name, compiled_model);
        if (!embed_mode_shared_context_serializer) {
          // Setup a deferred serializer for the first node which will be called after all nodes are compiled.
          embed_mode_shared_context_serializer = [node = std::move(node), io_mapping = std::move(io_mapping), this, &compiled, &ep_context_nodes, i]() mutable -> OrtStatus* {
            std::stringstream ss;
            if (compiled) {
              shared_context_->Serialize(ss);
            }
            node.main_context = 1;
            node.ep_cache_context = std::move(ss).str();
            ep_context_nodes[i] = node.CreateNode(io_mapping).release();
            return nullptr;
          };
          return nullptr;  // Defer serialization/node creation
        }
        node.main_context = 0;
        ep_context_nodes[i] = node.CreateNode(io_mapping).release();
      } else {
        node.main_context = 1;
        node.ep_cache_context = ToUtf8String(shared_context_->GetBinPath().filename());
        shared_context_->AddNativeBlob(node.partition_name, compiled_model);
        ep_context_nodes[i] = node.CreateNode(io_mapping).release();
      }
      return nullptr;
    };

    auto build_init_params = [&](const OvDevice& device) {
      auto configs = options_.provider_options_.GetDeviceConfig(device);
      if (ov_profiling_enabled_) {
        configs[ov::enable_profiling.name()] = true;
      }
      if (options_.ep_ctx_.weightless_) {
        configs[ov::enable_weightless.name()] = true;
      }

      WeightlessSourceModel weightless_source{
          .buffer = options_.ep_ctx_.source_model_buffer_,
          .file_path = options_.ep_ctx_.source_model_path_,
      };

      const bool reuse_const_data = device.IsDeviceType("GPU") ||
                                    (device.IsDeviceType("NPU") && !options_.ep_ctx_.weightless_);

      return OvComputeInitParams{
          .ov_device = device,
          .configs = std::move(configs),
          .model_info = model_info,
          .transformations = BuildTransformationPipeline(paths, output_node_name),
          .onnx_fe_manager = onnx_fe_manager_,
          .weightless_source = std::move(weightless_source),
          .shared_context = options_.ep_ctx_.share_ ? shared_context_ : nullptr,
          .reuse_const_data = reuse_const_data};
    };

    if (num_nodes == 1) {
      // Checking for EP context node
      bool is_ep_context = EpContextNode::IsEpContextNode(nodes[0]);
      if (is_ep_context) {
        // Running an EP context model
        EpContextNode ep_context_node(nodes[0], output_model_path, model_path);

        const bool is_native_ep_context = ep_context_node.private_fields_.type == EpContextNode::EpContextType::Native;
        ep_context_node.private_fields_.shared_context = shared_context_;

        OVEP_RETURN_IF_ERROR(ov_compute->Init(build_init_params(ov_device_), std::move(ep_context_node)));
        OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));
        compiled |= !is_native_ep_context;
        node_compute_infos[i] = ov_compute.release();
        continue;
      }
    }

    // Compiling a regular model
    ModelInitializerInfo model_initializer_info;
    OVEP_RETURN_IF_ERROR(model_initializer_info.Init(ort_graph));
    model_info.has_external_weights = !model_initializer_info.initializers_.empty();

    // Weightless only supports external constants in OV 2026.4
    if (options_.ep_ctx_.weightless_ && !model_info.has_external_weights) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "Weightless mode requires a model with external weights. "
                                  "Internal-only weightless is not supported.");
    }

    // Record the external weights file in the bin
    if (options_.ep_ctx_.enable_ && options_.ep_ctx_.weightless_) {
      shared_context_->SetWeightsFile(model_initializer_info.initializers_.begin()->second.external_file_path);
    }

    if (options_.ep_ctx_.share_) {
      AddSharableWeights(model_initializer_info, shared_context_);
    }

    OVEP_RETURN_IF_ERROR(ov_compute->Init(build_init_params(ov_device_), ort_graph));
    compiled = true;
    OVEP_RETURN_IF_ERROR(try_export_ep_context(*ov_compute, std::move(io_mapping)));

    node_compute_infos[i] = ov_compute.release();
  }

  // populate our computes cache.
  for (size_t i = 0; i < count; ++i) {
    computes_.insert(static_cast<OvComputeInfo*>(node_compute_infos[i]));
  }

  if (options_.ep_ctx_.enable_ && compiled) {
    if (embed_mode_shared_context_serializer) {
      // Embed mode: blob is embedded in the ONNX model itself; write callback does not apply.
      OVEP_RETURN_IF_ERROR(embed_mode_shared_context_serializer());
    } else {
      const bool finalize = options_.ep_ctx_.stop_ || !options_.ep_ctx_.share_;
      if (ep_ctx_write_fn_) {
        if (finalize) {
          std::string blob = shared_context_->SerializeToString();
          const std::string file_name = ToUtf8String(shared_context_->GetBinPath().filename());
          // Throws on host callback failure.
          ep_ctx_write_fn_(file_name, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(blob.data()), blob.size()));
        } else {
          // share=true, stop=false: blob is still accumulating; write is deferred to the stop
          // session. If the stop session runs without ep_ctx_write_fn_ set, the blob will be
          // written to disk instead via Serialize() — or silently lost if the path is unset.
          ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE,
                      "EPContext write callback deferred: sharing session, waiting for stop session to write blob");
        }
      } else {
        if (options_.ep_ctx_.stop_ && !shared_context_->GetBinPath().empty()) {
          ORT_CXX_LOGF(logger_, ORT_LOGGING_LEVEL_VERBOSE,
                       "EPContext stop session: writing blob to disk (no write callback registered): %s",
                       ToUtf8String(shared_context_->GetBinPath()).c_str());
        }
        shared_context_->Serialize();
      }
    }
    if (options_.ep_ctx_.stop_) {
      shared_context_manager_->ClearActiveSharedContext();
    }
  }

  return nullptr;
}

void OpenVINOEpPlugin::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                               size_t num_node_compute_infos) {
  // Clean up any compute info objects
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    auto* compute_info = static_cast<OvComputeInfo*>(node_compute_infos[i]);
    computes_.erase(compute_info);
    delete compute_info;
  }
}

OrtStatus* OpenVINOEpPlugin::CreateProfiler(OrtEpProfilerImpl** profiler) {
  OVEP_RETURN_IF(!profiler, ort_api, "profiler output cannot be null");

  auto profiler_impl = new OVEpProfilerImpl(logger_);

  // Set callback to synchronize profiling state with EP. Since OpenVINOEpPlugin's lifetime
  // is longer than the profiler's (profiler is released before EP is destroyed), capturing
  // 'this' by reference in the lambda is safe.
  profiler_impl->SetProfilingStateCallback(
      [this](bool profiling_enabled) { ov_profiling_enabled_ = profiling_enabled; });

  *profiler = profiler_impl;
  return nullptr;
}
