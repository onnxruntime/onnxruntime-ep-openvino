// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <format>
#include <limits>

#include "ort_api_version.h"
#include "ov_compute.h"
#include "ov_ep_context.h"
#include "ov_ep_profiler.h"
#include "ov_utils.h"
#include "ov_version.h"
#include "ov_plugin_utils.h"

#include "onnxruntime/onnxruntime_session_options_config_keys.h"
#include "openvino/runtime/properties.hpp"

using namespace onnxruntime::openvino_ep_plugin;

namespace onnxruntime {
namespace openvino_ep_plugin {

// Minimum ORT API version exposing KernelContext_GetPreallocatedOutput.
constexpr uint32_t kUserProvidedOutputApiVersion = 30;

OvComputeInfo::OvComputeInfo(ov::Core& ov_core, Ort::Logger logger) : OrtNodeComputeInfo{}, ov_core_(ov_core), logger_(logger) {
  ort_version_supported = ort_runtime_api_version;
  OrtNodeComputeInfo::CreateState = OVEP_API_ENTRY(OvComputeInfo, CreateState);
  OrtNodeComputeInfo::Compute = OVEP_API_ENTRY(OvComputeInfo, Compute);
  OrtNodeComputeInfo::ReleaseState = OVEP_API_ENTRY(OvComputeInfo, ReleaseState);
}

OvComputeInfo::~OvComputeInfo() {
  dynamic_output_pool_.reset();
  user_output_pool_.reset();

  if (compiled_model_) {
    try {
      compiled_model_.release_memory();
    } catch (...) {
      ORT_CXX_LOG_NOEXCEPT(logger_, ORT_LOGGING_LEVEL_WARNING,
                           "Failed to release OpenVINO compiled model memory.");
    }
  }
}

// For weightless models, OV needs a hint to resolve weights - either the weights
// file itself (ov::weights_path) or the source model that owns them (ov::hint::model).
// The source model may come from a buffer or a path; the buffer takes precedence when both are set.
OrtStatus* OvComputeInfo::AddWeightlessImportConfigs(const OvComputeInitParams& params,
                                                     const std::filesystem::path& weights_file,
                                                     ov::AnyMap& configs) {
  const auto& source = params.weightless_source;
  const bool has_source_model = !source.buffer.empty() || !source.file_path.empty();

  if (!has_source_model) {
    // ORT points the model path at session.model_external_initializers_file_folder_path when the app
    // sets it, and at the EPCtx model otherwise, so either way this is where the weights are
    const std::filesystem::path weights_dir = params.model_info.model_path.parent_path();
    const std::filesystem::path resolved_weights = weights_dir / weights_file;

    OVEP_RETURN_STATUS_IF(!std::filesystem::exists(resolved_weights), ORT_INVALID_ARGUMENT,
                          std::format("Weightless import cannot find the weights at '{}'. Set {} to the source "
                                      "model, or point {} at the weights directory together with {}.",
                                      ToUtf8String(resolved_weights),
                                      kOrtSessionOptionEpContextSourceModelPath,
                                      kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                      kOrtSessionOptionEpContextFilePath)
                              .c_str());
    configs[ov::weights_path.name()] = ToUtf8String(resolved_weights);
    return nullptr;
  }

  std::shared_ptr<ov::Model> ov_model;

  // Read the source model bytes into a string and hand a single stream to the ONNX frontend so that
  // both internal- and external-weight source models take one read path. A buffer has no directory of
  // its own, so ORT points model_info.model_path at model_external_initializers_file_folder_path; a file
  // resolves its own external initializers relative to its own path.
  std::string source_model;
  std::filesystem::path source_model_path;
  if (!source.buffer.empty()) {
    source_model.assign(reinterpret_cast<const char*>(source.buffer.data()), source.buffer.size());
    source_model_path = params.model_info.model_path;
  } else {
    OVEP_RETURN_STATUS_IF(!std::filesystem::exists(source.file_path), ORT_INVALID_ARGUMENT,
                          std::format("{} does not exist: {}", kOrtSessionOptionEpContextSourceModelPath,
                                      ToUtf8String(source.file_path))
                              .c_str());

    std::ifstream file(source.file_path, std::ios::binary);
    OVEP_RETURN_STATUS_IF(!file, ORT_INVALID_ARGUMENT,
                          std::format("Failed to open {}: {}", kOrtSessionOptionEpContextSourceModelPath,
                                      ToUtf8String(source.file_path))
                              .c_str());
    source_model.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    source_model_path = source.file_path;
  }

  OVEP_TRACE(logger_, "ReadModelWithExternalWeightSupport(source_model_path=%s, bytes=%zu)",
             ToUtf8String(source_model_path).c_str(), source_model.size());
  OVEP_RETURN_IF_ERROR(
      ReadModelWithExternalWeightSupport(std::move(source_model), params, source_model_path, ov_model));

  configs[ov::hint::model.name()] = std::shared_ptr<const ov::Model>(std::move(ov_model));
  return nullptr;
}

OrtStatus* OvComputeInfo::ImportNativeEpContext(const OvComputeInitParams& params, EpContextNode&& ep_context_node) {
  OVEP_TRACE(logger_);

  if (!params.transformations.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Model is native ep ctx. Model transformation functions will be skipped.");
  }

  auto& shared_context = ep_context_node.private_fields_.shared_context;
  OVEP_ENFORCE(shared_context, "Missing shared_context: required to load native blob for partition");

  try {
    const ov::Tensor model_tensor = shared_context->GetNativeBlob(ep_context_node.partition_name);

    auto configs = params.configs;

    // The external weights location is recorded in the bin only for weightless exports, so an empty
    // path means the blob carries its own weights
    const std::filesystem::path weights_file = shared_context->GetWeightsFile();
    if (!weights_file.empty()) {
      OVEP_RETURN_IF_ERROR(AddWeightlessImportConfigs(params, weights_file, configs));
    }

    {
      OVEP_TRACE(logger_, "ov::Core::import_model(tensor, device=%s)", params.ov_device.c_str());
      compiled_model_ = ov_core_.import_model(model_tensor, params.ov_device, configs);
    }
  } catch (const std::exception& e) {
    if (ShouldRecompile(e.what())) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR,
                  "EPCtx model is incompatible, ImportModel() failed");
    }
    return ort_api.CreateStatus(ORT_INVALID_GRAPH, OVEP_ERROR_STR(e.what()).c_str());
  }
  return nullptr;
}

OrtStatus* OvComputeInfo::Init(OvComputeInitParams&& params, EpContextNode&& ep_context_node) {
  OVEP_TRACE(logger_, "ep_context");

  const OnnxIOMapping& io_mapping = params.model_info.io_mapping;

  switch (ep_context_node.private_fields_.type) {
    case EpContextNode::EpContextType::Native:
      OVEP_RETURN_IF_ERROR(ImportNativeEpContext(params, std::move(ep_context_node)));
      break;

    case EpContextNode::EpContextType::OV_IR:
      if (ep_context_node.embed_mode != 0) {
        // To support this we would need to save the weights location somewhere, or have a schema for embedding the weights along with the IR.
        return Ort::Status("Epctx with OVIR must use embed_mode == 0", ORT_INVALID_ARGUMENT).release();
      } else {
        const std::filesystem::path ep_ctx_path = ep_context_node.GetEpContextPath();
        if (params.transformations.empty()) {
          {
            OVEP_TRACE(logger_, "ov::Core::compile_model(path=%s, device=%s)", ToUtf8String(ep_ctx_path).c_str(), params.ov_device.c_str());
            compiled_model_ = ov_core_.compile_model(ep_ctx_path, params.ov_device, params.configs);
          }
        } else {
          std::filesystem::path bin_path = ep_ctx_path.parent_path() / ep_ctx_path.stem();
          bin_path += ".bin";
          std::shared_ptr<ov::Model> ov_model;
          {
            OVEP_TRACE(logger_, "ov::Core::read_model(path=%s)", ToUtf8String(ep_ctx_path).c_str());
            ov_model = ov_core_.read_model(ep_ctx_path, bin_path);
          }
          OVEP_RETURN_IF_ERROR(params.transformations.ApplyTransformations(ov_model));
          {
            OVEP_TRACE(logger_, "ov::Core::compile_model(model, device=%s)", params.ov_device.c_str());
            compiled_model_ = ov_core_.compile_model(ov_model, params.ov_device, params.configs);
          }
        }
      }
      break;

    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  InitFooter(io_mapping, std::move(params.transformations), params.ov_device, params.shared_context);
  return nullptr;
}

OrtStatus* OvComputeInfo::Init(OvComputeInitParams&& params, std::string&& model) {
  return InitCommon(std::move(params), std::move(model));
}

OrtStatus* OvComputeInfo::Init(OvComputeInitParams&& params, const OrtGraph* graph) {
  return InitCommon(std::move(params), graph);
}

template <typename T>
OrtStatus* OvComputeInfo::InitCommon(OvComputeInitParams&& params, T&& model) {
  OVEP_TRACE(logger_);

  const OnnxIOMapping& io_mapping = params.model_info.io_mapping;
  std::shared_ptr<ov::Model> ov_model;
  {
    OVEP_TRACE(logger_, "ReadModelWithExternalWeightSupport(path=%s)", ToUtf8String(params.model_info.model_path).c_str());
    OVEP_RETURN_IF_ERROR(ReadModelWithExternalWeightSupport(std::move(model), params, params.model_info.model_path, ov_model));
  }
  OVEP_RETURN_IF_ERROR(params.transformations.ApplyTransformations(ov_model));
  {
    OVEP_TRACE(logger_, "ov::Core::compile_model(model, device=%s)", params.ov_device.c_str());
    compiled_model_ = ov_core_.compile_model(ov_model, params.ov_device, params.configs);
  }

  InitFooter(io_mapping, std::move(params.transformations), params.ov_device, params.shared_context);
  return nullptr;
}

void OvComputeInfo::InitFooter(const OnnxIOMapping& io_mapping,
                               TransformationPipeline&& transformations,
                               const OvDevice& ov_device,
                               const std::shared_ptr<SharedContext>& shared_context) {
  if (!ov_device.IsVirtual()) {
    infer_request_initializer_ = [transformations = std::move(transformations)](InferRequestPool::OVInferRequestPtr& infer_request) {
      transformations.InitializeInferRequest(infer_request->ov());
    };

    dynamic_output_pool_ = std::make_unique<InferRequestPool>(compiled_model_, 1, infer_request_initializer_);
  }
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{}, shared_context);
}

OrtStatus* OvComputeInfo::TryInitPreallocatedOutputBindings(OrtKernelContext* kernel_context,
                                                            std::vector<UserOutputBinding>& bindings,
                                                            bool& user_bindings_compatible) const {
  user_bindings_compatible = false;
  bindings.clear();

  const auto& network_outputs = onnx_to_ov_bindings_->network_outputs_;
  if (network_outputs.empty() || ort_runtime_api_version < kUserProvidedOutputApiVersion) {
    return nullptr;
  }

  std::vector<OrtValue*> user_outputs(network_outputs.size(), nullptr);
  OrtStatus* query_status = nullptr;
  OrtVersionGate<kUserProvidedOutputApiVersion>(&ort_api, [&](auto* api) {
    for (size_t i = 0; i < network_outputs.size(); ++i) {
      // An out-of-range index returns a status rather than null, so a null below unambiguously
      // means "ORT owns this output".
      query_status = api->KernelContext_GetPreallocatedOutput(kernel_context, network_outputs[i].onnx_index,
                                                              &user_outputs[i]);
      if (query_status != nullptr) {
        return;
      }
    }
  });
  OVEP_RETURN_IF_ERROR(query_status);

  bindings.reserve(network_outputs.size());

  // Binding is all-or-nothing: one unusable output sends the whole invocation down the copy path,
  // so the first rejection can stop the loop. None of the accessors below can throw for a concrete
  // allocated tensor, but a throw is just one more reason to fall back.
  for (size_t i = 0; i < network_outputs.size(); ++i) {
    if (user_outputs[i] == nullptr) {
      break;
    }

    const auto& output_info = network_outputs[i];
    Ort::UnownedValue value{user_outputs[i]};
    const auto tensor_info = value.GetTensorTypeAndShapeInfo();
    const auto element_type = tensor_info.GetElementType();

    // Strings are rejected separately because they cannot be wrapped in an ov::Tensor over caller memory.
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      break;
    }

    auto shape = tensor_info.GetShape();
    bindings.push_back(UserOutputBinding{&output_info,
                                         user_outputs[i],
                                         shape,
                                         ov::Tensor(output_info.type, ParameterShape(shape), value.GetTensorMutableRawData())});
  }

  if (bindings.size() != network_outputs.size()) {
    // Only use user mappings if all the outputs are user bound.
    bindings.clear();
    return nullptr;
  }

  user_bindings_compatible = true;
  return nullptr;
}

InferRequestPool& OvComputeInfo::GetPreallocatedOutputInferPool() {
  std::lock_guard<std::mutex> lock(user_output_pool_mutex_);
  if (!user_output_pool_) {
    OVEP_ENFORCE(dynamic_output_pool_ != nullptr && infer_request_initializer_,
                 "User-output pool cannot be initialized before the dynamic-output pool");
    user_output_pool_ = std::make_unique<InferRequestPool>(compiled_model_, 1, infer_request_initializer_);
  }
  return *user_output_pool_;
}

OrtStatus* OvComputeInfo::Compute(void* /*compute_state*/,
                                  OrtKernelContext* kernel_context) {
  OVEP_RETURN_IF(!dynamic_output_pool_, ort_api, "Infer request pool not initialized for virtual devices.");

  Ort::KernelContext context(kernel_context);

  if (onnx_to_ov_bindings_->has_dynamic_io_) {
    // Dynamic shape inference

    // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
    // tensors instead of binding them to the infer request directly.

    // If the user provided output buffers and ORT supports querying them, we can avoid the copy. Otherwise fallback to copy.

    std::vector<UserOutputBinding> user_output_bindings;
    bool use_prealloc_bindings = false;
    OVEP_RETURN_IF_ERROR(TryInitPreallocatedOutputBindings(kernel_context, user_output_bindings, use_prealloc_bindings));

    // Use the user-output pool only when every output is caller-owned. OpenVINO has no unbind
    // operation, so mixing caller-bound outputs with plugin-managed outputs could leave a pooled
    // request contaminated with an external tensor on a later invocation.
    auto* request_pool = use_prealloc_bindings ? &GetPreallocatedOutputInferPool() : dynamic_output_pool_.get();
    auto guarded_infer_req = request_pool->getRequest();
    auto& infer_request = guarded_infer_req.infer_request_;

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      // Set the input shape based on the input tensor from ort
      auto tensor = context.GetInput(input_info.onnx_index);
      auto&& ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
      auto input_shape = ParameterShape(ort_shape);

      // Bound to the ORT shape; the OpenVINO element type was checked against the
      // ONNX declaration at binding construction, so there is nothing to re-check.
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_shape,
                               const_cast<void*>(tensor.GetTensorRawData()));
    }

    // Optionally: Bind outputs if there's user provided output buffers.
    if (use_prealloc_bindings) {
      for (const auto& binding : user_output_bindings) {
        infer_request->SetTensor(binding.output_info->name,
                                 binding.output_info->type,
                                 binding.tensor.get_shape(),
                                 const_cast<void*>(binding.tensor.data()));
      }
    }

    // Run Inference
    {
      OVEP_TRACE(logger_, "ov::InferRequest::infer() [dynamic_io]");
      infer_request->Infer();
    }
    OVEpProfilerImpl::CaptureInferRequestProfiling(infer_request->ov());

    // Copy outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      auto ov_tensor = infer_request->ov().get_tensor(output_info.name);
      auto&& output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
      auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);

      if (use_prealloc_bindings) {
        continue;  // We don't need to copy, but we still want to call context.GetOutput to get ORT to validate the output shape.
      }

      if (ov_tensor.get_element_type() == ov::element::string) {
        OVEP_RETURN_IF(ov_tensor.get_size() != ort_tensor.GetTensorTypeAndShapeInfo().GetElementCount(),
                       ort_api,
                       std::format("Output tensor size mismatch for {}", output_info.name).c_str());

        const auto* ov_strings = ov_tensor.data<std::string>();
        for (size_t i = 0; i < ov_tensor.get_size(); ++i) {
          ort_tensor.FillStringTensorElement(ov_strings[i].c_str(), i);
        }
      } else {
        OVEP_RETURN_IF(ov_tensor.get_byte_size() != ort_tensor.GetTensorSizeInBytes(),
                       ort_api,
                       std::format("Output tensor size mismatch for {}", output_info.name).c_str());

        // Zero-element output: data() may be null, and memcpy with a null pointer is UB even at
        // size 0. Nothing to copy, so skip.
        if (ov_tensor.get_byte_size() == 0) {
          continue;
        }

        std::memcpy(ort_tensor.GetTensorMutableRawData(),
                    ov_tensor.data(),
                    ov_tensor.get_byte_size());
      }
    }
  } else {
    // Static shape inference.

    auto guarded_infer_req = dynamic_output_pool_->getRequest();
    auto& infer_request = guarded_infer_req.infer_request_;

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      auto input_tensor = context.GetInput(input_info.onnx_index);
      auto&& ort_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

      // Bind the ORT shape, not the OpenVINO parameter shape. Element type was
      // verified against the ONNX declaration at binding construction; binding the
      // caller's shape lets OpenVINO reject a tensor that doesn't fit the static
      // parameter (e.g. an undersized reshape_input/free_dim_override input) instead
      // of over-reading the allocation.
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               ParameterShape(ort_shape),
                               const_cast<void*>(input_tensor.GetTensorRawData()));
    }

    // Bind outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      auto output_tensor = context.GetOutput(output_info.onnx_index, output_info.shape);
      infer_request->SetTensor(output_info.name,
                               output_info.type,
                               output_info.shape,
                               output_tensor.GetTensorMutableRawData());
    }

    // Run Inference
    {
      OVEP_TRACE(logger_, "ov::InferRequest::infer() [static_io]");
      infer_request->Infer();
    }
    OVEpProfilerImpl::CaptureInferRequestProfiling(infer_request->ov());
  }

  return nullptr;
}

OrtStatus* OvComputeInfo::SetWorkloadType(const std::string& workload_type) {
  if (compiled_model_) {
    try {
      OVEP_TRACE(logger_, "ov::CompiledModel::set_property(workload_type=%s)", workload_type.c_str());
      compiled_model_.set_property(ov::workload_type(workload_type));
    } catch (const std::exception& e) {
      return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                  std::format("set_property(ov::workload_type(%s)) failed. Details: %s", workload_type, e.what()).c_str());
    }
  } else {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "No compiled model to set workload_type on");
  }
  return nullptr;
}

OrtStatus* OvComputeInfo::CreateState(OrtNodeComputeContext* compute_context,
                                      void** compute_state) {
  // Dummy implementation: set compute_state to nullptr
  (void)compute_context;
  *compute_state = nullptr;
  return nullptr;
}

void OvComputeInfo::ReleaseState(void* compute_state) {
  // Dummy implementation: do nothing
  (void)compute_state;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
