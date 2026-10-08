// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <memory>
#include <functional>
#include <mutex>
#include <span>
#include <sstream>
#include <optional>
#include <gsl/narrow>

#include "ov_constants.h"
#include "ov_provider.h"
#include "ov_onnx_delegate.h"
#include "ov_device.h"

#include "openvino/runtime/core.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

using string_index_map_t = std::unordered_map<std::string, uint32_t>;

class WrappedInferRequest {
  struct ov_tensor_data_t {
    std::unique_ptr<ov::Tensor> tensor_ptr;
    const void* ort_ptr{nullptr};
  };

 protected:
  ov::InferRequest ov_inf_req_;
  std::unordered_map<std::string, ov_tensor_data_t> bindings_cache_;

 public:
  // Set tensor call infer req tensor if ort_ptr differs from last set ptr.
  void SetTensor(const std::string& name, const ov::element::Type& type, const ov::Shape& shape, void* ort_ptr) {
    auto& cached_binding = bindings_cache_[name];
    if (cached_binding.ort_ptr != ort_ptr ||
        !cached_binding.tensor_ptr ||
        cached_binding.tensor_ptr->get_shape() != shape) {
      cached_binding.tensor_ptr.reset();
      auto ov_tensor = std::make_unique<ov::Tensor>(type, shape, ort_ptr);
      ov_inf_req_.set_tensor(name, *ov_tensor);
      cached_binding = {std::move(ov_tensor), ort_ptr};
    }
  }

  void Infer() { ov_inf_req_.infer(); }
  ov::InferRequest& ov() { return ov_inf_req_; }
  explicit WrappedInferRequest(ov::InferRequest&& obj) : ov_inf_req_(std::move(obj)) {}
  // virtual void RewindKVCache(size_t index) {}
};

class InferRequestPool {
 public:
  using OVInferRequestPtr = std::unique_ptr<WrappedInferRequest>;
  struct GuardedInferReq {
    OVInferRequestPtr infer_request_;
    GuardedInferReq(InferRequestPool& queue, OVInferRequestPtr&& infer_req) : infer_request_(std::move(infer_req)), queue_(queue) {}
    ~GuardedInferReq() { queue_.putIdleRequest(std::move(infer_request_)); }

    // Movable but not copyable
    OVEP_DISABLE_COPY(GuardedInferReq);
    GuardedInferReq(GuardedInferReq&&) = default;
    GuardedInferReq& operator=(GuardedInferReq&&) = default;

   private:
    InferRequestPool& queue_;
    friend class InferRequestPool;
  };

  InferRequestPool(ov::CompiledModel& model, size_t initial_size, std::function<void(OVInferRequestPtr&)> initializer) : compile_model_(model), initializer_(std::move(initializer)) {
    for (size_t id = 0; id < initial_size; id++) {
      infer_requests_.emplace_back(createInferRequest());
    }
  }
  virtual ~InferRequestPool() = default;

  GuardedInferReq getRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (infer_requests_.empty()) {
      infer_requests_.emplace_back(createInferRequest());
    }
    auto request = std::move(infer_requests_.back());
    infer_requests_.pop_back();
    return GuardedInferReq(*this, std::move(request));
  }

  template <typename Func>
  void forEachIdleRequest(Func&& func) {
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& infer_request : infer_requests_) {
      func(infer_request);
    }
  }

 private:
  void putIdleRequest(OVInferRequestPtr&& infer_request) {
    if (infer_request) {
      std::unique_lock<std::mutex> lock(_mutex);
      infer_requests_.emplace_back(std::move(infer_request));
    }
  }

  OVInferRequestPtr createInferRequest() {
    auto infer_request = std::make_unique<WrappedInferRequest>(std::move(compile_model_.create_infer_request()));
    initializer_(infer_request);
    return infer_request;
  }

 private:
  std::mutex _mutex;
  std::vector<OVInferRequestPtr> infer_requests_;
  ov::CompiledModel& compile_model_;
  std::function<void(OVInferRequestPtr&)> initializer_;
};

struct ParameterShape {
  using ort_shape_t = std::vector<int64_t>;

  static ov::PartialShape ToOvPartialShape(const ort_shape_t& ort_shape) {
    std::vector<ov::Dimension> ov_shape(ort_shape.size());
    std::transform(ort_shape.begin(), ort_shape.end(), ov_shape.begin(), [](int64_t dim) {
      return dim == -1 ? ov::Dimension::dynamic() : ov::Dimension(dim);
    });
    return ov::PartialShape(ov_shape);
  }

  static ort_shape_t
  ToOrtShape(const ov::PartialShape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return dim.is_dynamic() ? -1 : dim.get_length();
    });
    return ort_shape;
  }

  static ort_shape_t ToOrtShape(const ov::Shape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return gsl::narrow<int64_t>(dim);
    });
    return ort_shape;
  }

  operator ov::Shape() const { return ov_.get_shape(); }
  operator const ov::PartialShape&() const { return ov_; }
  operator const ort_shape_t&() const { return ort_; }
  ort_shape_t& ort() { return ort_; }

  explicit ParameterShape(const ort_shape_t& ort_shape) : ort_(ort_shape), ov_(ToOvPartialShape(ort_shape)) {}
  explicit ParameterShape(const ov::PartialShape& ov_partial_shape) : ort_(ToOrtShape(ov_partial_shape)), ov_(ov_partial_shape) {}

 private:
  ort_shape_t ort_;
  ov::PartialShape ov_;
};

struct OnnxIOMapping {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  // ONNX-declared element type and shape per I/O, aligned with the name vectors.
  // Used once at binding construction to reject a compiled model whose OpenVINO
  // parameters disagree with the graph the caller declared. Dynamic dims are
  // stored as -1; a non-tensor I/O leaves the type ov::element::dynamic.
  std::vector<ov::element::Type> input_types;
  std::vector<ov::element::Type> output_types;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  OrtStatus* Init(const OrtApi& ort_api, const OrtGraph& graph) {
    std::vector<const OrtValueInfo*> io;
    size_t num_elements = 0;

    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumInputs(&graph, &num_elements));
    io.resize(num_elements);
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetInputs(&graph, io.data(), io.size()));
    OVEP_RETURN_IF_ERROR(Populate(ort_api, io, input_names, input_types, input_shapes));

    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetNumOutputs(&graph, &num_elements));
    io.resize(num_elements);
    OVEP_RETURN_IF_ERROR(ort_api.Graph_GetOutputs(&graph, io.data(), io.size()));
    OVEP_RETURN_IF_ERROR(Populate(ort_api, io, output_names, output_types, output_shapes));
    return nullptr;
  }

 private:
  static OrtStatus* Populate(const OrtApi& ort_api,
                             std::vector<const OrtValueInfo*>& io_nodes,
                             std::vector<std::string>& names_vec,
                             std::vector<ov::element::Type>& types_vec,
                             std::vector<std::vector<int64_t>>& shapes_vec) {
    for (const auto& value_info : io_nodes) {
      bool is_required_graph_input = false;
      OVEP_RETURN_IF_ERROR(ort_api.ValueInfo_IsRequiredGraphInput(value_info, &is_required_graph_input));

      bool is_graph_output = false;
      OVEP_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output));

      const char* name = nullptr;
      OVEP_RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));

      if (!((is_required_graph_input || is_graph_output) && name)) {
        continue;
      }
      names_vec.emplace_back(name);

      // Declared type/shape. Non-tensor I/O leaves an undefined type and empty
      // shape, which the binding check treats as "no constraint".
      ov::element::Type onnx_type = ov::element::dynamic;
      std::vector<int64_t> onnx_dims;
      const OrtTypeInfo* type_info = nullptr;
      OVEP_RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(value_info, &type_info));
      const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
      Ort::Status cast_status{ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info)};
      if (cast_status.IsOK() && tensor_info) {
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        OVEP_RETURN_IF_ERROR(ort_api.GetTensorElementType(tensor_info, &et));
        onnx_type = ort_tensor_meta_info::ConvertONNXTypeToOV(et);
        size_t dim_count = 0;
        OVEP_RETURN_IF_ERROR(ort_api.GetDimensionsCount(tensor_info, &dim_count));
        onnx_dims.resize(dim_count);
        if (dim_count > 0) {
          OVEP_RETURN_IF_ERROR(ort_api.GetDimensions(tensor_info, onnx_dims.data(), dim_count));
        }
      }
      types_vec.emplace_back(onnx_type);
      shapes_vec.emplace_back(std::move(onnx_dims));
    }
    return nullptr;
  }
};

struct ParameterInfo {
  std::string name;
  uint32_t ov_index;
  uint32_t onnx_index;
  ov::element::Type type;
  ParameterShape shape;
};

struct UserOutputBinding {
  const ParameterInfo* output_info{};
  OrtValue* ort_value{};
  std::vector<int64_t> ort_shape;
  ov::Tensor tensor;
};

struct SessionContext {
  bool enable_causallm = false;
};

struct OnnxToOvNetworkBindings {
  std::vector<ParameterInfo> network_outputs_;
  std::vector<ParameterInfo> network_inputs_;
  bool has_dynamic_io_ = false;

  inline static const std::array special_io_names_{
      "beam_idx",
      "past_key_values",
      "present",
  };

  inline static const std::array stateful_internal_input_names_{
      "beam_idx",
      "src_idx",
      "dst_idx",
  };

  OnnxToOvNetworkBindings(const ov::CompiledModel& exec_network,
                          const OnnxIOMapping& io_mapping,
                          const SessionContext& session_context,
                          const std::shared_ptr<SharedContext>& shared_context = nullptr) {
    auto populate = [&](auto& input_output_map, const std::vector<std::string>& onnx_io,
                        const std::vector<ov::element::Type>& onnx_types,
                        const std::vector<std::vector<int64_t>>& onnx_shapes,
                        const auto& ov_parameters) {
      for (uint32_t i = 0; i < onnx_io.size(); i++) {
        const auto& onnx_name = onnx_io[i];
        const uint32_t onnx_param_index = i;

        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });
        bool matched_names = it != ov_parameters.end();

        // For Stateful Model Compilation, the ONNX model includes KV cache (past/present) tensors.
        // However, these tensors are internally converted to a stateful representation, which removes them.
        // It's also possible that the onnx model does not contain tensors such as beam_idx, whereas our converted
        // stateful representation has introduced these new tensors, creating a name mismatch (matched_names=false).
        // So, if there is a name mismatch, or the name matches our special io list, we simply continue processing
        // here to prevent runtime exceptions.
        if (session_context.enable_causallm) {
          if (!matched_names ||
              std::any_of(special_io_names_.begin(), special_io_names_.end(),
                          [&onnx_name](const std::string& name) { return onnx_name.find(name) != std::string::npos; })) {
            // This case also requires dynamic shape inference, so we'll mark the bindings as dynamic.
            has_dynamic_io_ = true;
            continue;
          }
        }

        OVEP_ENFORCE(matched_names,
                     "Input names mismatch between OpenVINO and ONNX. ", onnx_name,
                     " doesn't exist in the list of OpenVINO input tensor names");

        uint32_t ov_param_index = gsl::narrow<uint32_t>(std::distance(ov_parameters.begin(), it));

        const auto& shape = ov_parameters[ov_param_index].get_partial_shape();
        auto type = ov_parameters[ov_param_index].get_element_type();
        ParameterInfo info{onnx_name, ov_param_index, onnx_param_index, type, ParameterShape{shape}};

        // Analyze shape dynamism and set flags
        if (!shape.is_static()) {
          has_dynamic_io_ = true;
        }

        // The compiled model is untrusted input: its OpenVINO parameter must not
        // be assumed to match the ONNX I/O the caller declared. Validate once here
        // (the model is fixed after import) instead of on every inference. A
        // dynamic ONNX dim (e.g. reshape_input/free_dim_override) can't be
        // equality-checked here; Compute() binds the ORT shape and OpenVINO rejects
        // a tensor that doesn't fit the static parameter.
        // Skipped for stateful models, whose I/O is rewritten by the KV transform.
        const std::vector<int64_t>& onnx_dims = onnx_shapes[i];
        const bool onnx_shape_static =
            onnx_types[i] != ov::element::dynamic &&
            std::none_of(onnx_dims.begin(), onnx_dims.end(), [](int64_t d) { return d < 0; });

        if (!session_context.enable_causallm) {
          OVEP_ENFORCE(onnx_types[i] == ov::element::dynamic || onnx_types[i] == type,
                       "Type mismatch between ONNX and OpenVINO for ", onnx_name);
          if (onnx_shape_static && shape.is_static()) {
            size_t onnx_count = 1;
            for (int64_t d : onnx_dims) {
              onnx_count *= static_cast<size_t>(d);
            }
            OVEP_ENFORCE(onnx_count == ov::shape_size(shape.get_shape()),
                         "I/O size mismatch between ONNX and OpenVINO for ", onnx_name);
          }
        }

        input_output_map.push_back(std::move(info));
      }
    };

    // Init inputs and outputs
    populate(network_inputs_, io_mapping.input_names, io_mapping.input_types, io_mapping.input_shapes, exec_network.inputs());
    populate(network_outputs_, io_mapping.output_names, io_mapping.output_types, io_mapping.output_shapes, exec_network.outputs());

    const auto& ov_inputs = exec_network.inputs();
    for (uint32_t ov_index = 0; ov_index < ov_inputs.size(); ++ov_index) {
      const bool is_mapped_input = std::any_of(network_inputs_.begin(), network_inputs_.end(),
                                               [ov_index](const ParameterInfo& input) { return input.ov_index == ov_index; });
      const bool is_shared_weight = shared_context && std::any_of(ov_inputs[ov_index].get_names().begin(),
                                                                  ov_inputs[ov_index].get_names().end(),
                                                                  [&shared_context](const std::string& name) {
                                                                    return shared_context->IsSharedWeight(name);
                                                                  });
      const bool is_stateful_internal_input = session_context.enable_causallm &&
                                              std::any_of(ov_inputs[ov_index].get_names().begin(),
                                                          ov_inputs[ov_index].get_names().end(),
                                                          [](const std::string& name) {
                                                            return std::find(stateful_internal_input_names_.begin(),
                                                                             stateful_internal_input_names_.end(),
                                                                             name) != stateful_internal_input_names_.end();
                                                          });
      OVEP_ENFORCE(is_mapped_input || is_shared_weight || is_stateful_internal_input,
                   "Input names mismatch between OpenVINO and ONNX. OpenVINO input has no ONNX binding.");
    }
  }
};

struct OvModelInfo {
  OnnxIOMapping io_mapping{};
  std::filesystem::path model_path{};
  bool has_external_weights{false};
};

struct EpContextNode;

// Import-time inputs for rebinding a weightless blob's weights
struct WeightlessSourceModel {
  std::span<const uint8_t> buffer{};  // SessionOptionsSetWeightlessSourceModelBuffer -> ov::hint::model
  std::filesystem::path file_path{};  // ov::hint::model
};

struct OvComputeInitParams {
  OvDevice ov_device;
  ov::AnyMap configs;
  OvModelInfo model_info;
  TransformationPipeline transformations;
  std::shared_ptr<ov::frontend::FrontEndManager> onnx_fe_manager;
  WeightlessSourceModel weightless_source{};
  std::shared_ptr<SharedContext> shared_context;
  bool reuse_const_data{false};
};

struct OvComputeInfo : OrtNodeComputeInfo {
  OvComputeInfo(ov::Core& ov_core, Ort::Logger logger);

  virtual ~OvComputeInfo();

  OrtStatus* CreateState(OrtNodeComputeContext* compute_context,
                         void** compute_state);
  virtual OrtStatus* Compute(void* compute_state,
                             OrtKernelContext* kernel_context);
  void ReleaseState(void* compute_state);

  virtual OrtStatus* Init(OvComputeInitParams&& params, EpContextNode&& ep_context_node);
  virtual OrtStatus* Init(OvComputeInitParams&& params, std::string&& model);
  virtual OrtStatus* Init(OvComputeInitParams&& params, const OrtGraph* graph);

  ov::CompiledModel& GetCompiledModel() { return compiled_model_; }
  const ov::CompiledModel& GetCompiledModel() const { return compiled_model_; }

  // SetDynamicOptions routines
  OrtStatus* SetWorkloadType(const std::string& workload_type);
  virtual OrtStatus* KVCacheRewind(const size_t& /*index*/) { return nullptr; };
  virtual OrtStatus* SetReorderKVCacheStatus(const std::vector<int32_t>& /*src_indices*/, const std::vector<int32_t>& /*dst_indices*/) {
    return nullptr;
  }

  Ort::Logger GetLogger() const { return logger_; }

 protected:
  OrtStatus* ImportNativeEpContext(const OvComputeInitParams& params, EpContextNode&& ep_context_node);

  OrtStatus* AddWeightlessImportConfigs(const OvComputeInitParams& params,
                                        const std::filesystem::path& weights_file,
                                        ov::AnyMap& configs);

  template <typename T>
  OrtStatus* InitCommon(OvComputeInitParams&& params, T&& model);

  void InitFooter(const OnnxIOMapping&,
                  TransformationPipeline&& transformations,
                  const OvDevice& ov_device,
                  const std::shared_ptr<SharedContext>& shared_context);

  // Query and validate all caller-provided output tensors. This helper never
  // allocates an ORT output. `ready` is false when the runtime is too old or
  // any required output is absent/incompatible; those cases select fallback.
  OrtStatus* TryInitPreallocatedOutputBindings(OrtKernelContext* kernel_context,
                                               std::vector<UserOutputBinding>& bindings,
                                               bool& ready) const;

  InferRequestPool& GetPreallocatedOutputInferPool();

  // Reads an ONNX model into an ov::Model.
  // The ONNX frontend resolves external initializers relative to model_path
  template <typename T>
  OrtStatus* ReadModelWithExternalWeightSupport(T&& model,
                                                const OvComputeInitParams& init_params,
                                                const std::filesystem::path& model_path,
                                                std::shared_ptr<ov::Model>& ov_model) {
    // Load ONNX frontend
    OVEP_ENFORCE(init_params.onnx_fe_manager, "ONNX FrontEndManager is not initialized");
    auto frontend = init_params.onnx_fe_manager->load_by_framework("onnx");
    OVEP_ENFORCE(frontend, "Failed to load ONNX Frontend");

    ov::AnyVector params;
    std::istringstream modelStringStream;
    if constexpr (std::is_same_v<T, std::string>) {
      modelStringStream.str(std::move(model));
      params = {static_cast<std::istream*>(&modelStringStream), ToUtf8String(model_path)};
    } else if constexpr (std::is_same_v<T, const OrtGraph*>) {
      auto model_dir = model_path.parent_path();
      auto delegate = get_delegate_instance(model, std::move(model_dir), logger_);
      OVEP_ENFORCE(delegate, "Failed to create ONNX GraphIterator delegate");

      params = {delegate, enable_mmap, init_params.reuse_const_data};
    } else {
      static_assert(always_false_v<T>);
    }

    // Load input model from your iterator
    auto inputModel = frontend->load(params);
    OVEP_ENFORCE(inputModel, "Failed to load ONNX model");

    // Convert to ov::Model
    ov_model = frontend->convert(inputModel);
    return nullptr;
  }

  ov::CompiledModel compiled_model_;
  const OrtApi& ort_api{Ort::GetApi()};
  ov::Core& ov_core_;
  std::unique_ptr<InferRequestPool> dynamic_output_pool_;
  std::unique_ptr<InferRequestPool> user_output_pool_;
  std::function<void(InferRequestPool::OVInferRequestPtr&)> infer_request_initializer_;
  mutable std::mutex user_output_pool_mutex_;
  std::unique_ptr<OnnxToOvNetworkBindings> onnx_to_ov_bindings_;
  Ort::Logger logger_;
};
}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
