// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

#include "ov_utils.h"

#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace openvino_ep_plugin {

// ============================================================================
// ort_tensor_meta_info: extends TensorMetaInfo to provide storage for the name
// ============================================================================
struct ort_tensor_meta_info : ov::frontend::onnx::TensorMetaInfo {
  using base = ov::frontend::onnx::TensorMetaInfo;

  ort_tensor_meta_info(std::string name = "") : base{}, name_(std::move(name)) {
    m_tensor_name = &name_;
  }

  explicit ort_tensor_meta_info(const Ort::ConstValueInfo value_info)
      : ort_tensor_meta_info(value_info == nullptr ? std::string{} : value_info.GetName()) {
    if (value_info == nullptr) return;

    const Ort::ConstTypeInfo type_info = value_info.TypeInfo();
    const bool is_tensor = type_info != nullptr && type_info.GetONNXType() == ONNX_TYPE_TENSOR;
    if (is_tensor) {
      m_element_type = ConvertONNXTypeToOV(type_info.GetTensorTypeAndShapeInfo().GetElementType());
    }

    if (value_info.IsConstantInitializer()) {
      fetch_initializer_data(value_info);
    }

    if (is_tensor) {
      const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
      apply_shape_and_size(tensor_info.GetShape(), m_element_type, m_is_raw);
    }
  }

  struct initializer_tag {};
  ort_tensor_meta_info(const Ort::ConstValueInfo value_info, const std::optional<initializer_tag>& initializer) : base{} {
    if (value_info == nullptr) ORT_CXX_API_THROW("Expected value_info to be non-null", ORT_EP_FAIL);

    name_ = value_info.GetName();
    m_tensor_name = &name_;

    const Ort::ConstTypeInfo type_info = value_info.TypeInfo();
    const bool is_tensor = type_info != nullptr && type_info.GetONNXType() == ONNX_TYPE_TENSOR;
    if (is_tensor) {
      m_element_type = ConvertONNXTypeToOV(type_info.GetTensorTypeAndShapeInfo().GetElementType());
    }

    if (initializer) {
      fetch_initializer_data(value_info);
    }

    if (is_tensor) {
      const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
      apply_shape_and_size(tensor_info.GetShape(), m_element_type, m_is_raw);
    }
  }

  ort_tensor_meta_info(const ort_tensor_meta_info& other) : base(other), name_(other.name_) {
    m_tensor_name = &name_;
  }

  ort_tensor_meta_info& operator=(const ort_tensor_meta_info& other) {
    static_cast<base&>(*this) = other;
    name_ = other.name_;
    m_tensor_name = &name_;
    return *this;
  }

  ort_tensor_meta_info(ort_tensor_meta_info&& other) noexcept : base(std::move(other)), name_(std::move(other.name_)) {
    m_tensor_name = &name_;
  }

  ort_tensor_meta_info& operator=(ort_tensor_meta_info&& other) noexcept {
    static_cast<base&>(*this) = std::move(other);
    name_ = std::move(other.name_);
    m_tensor_name = &name_;
    return *this;
  }

  ~ort_tensor_meta_info() = default;

  const std::string& name() const { return name_; }

  // Fills m_partial_shape from dims (negative = dynamic); sets m_tensor_data_size for static shapes.
  void apply_shape_and_size(const std::vector<int64_t>& dims,
                            const ov::element::Type& element_type,
                            const bool is_raw) {
    std::vector<ov::Dimension> ov_dims;
    ov_dims.reserve(dims.size());
    for (const int64_t d : dims) {
      ov_dims.emplace_back(d < 0 ? ov::Dimension() : ov::Dimension(d));
    }
    m_partial_shape = ov::PartialShape(std::move(ov_dims));

    const bool is_static = dims.empty() || std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d > 0; });
    if (is_static && !m_external_location) {  // m_tensor_data_size is already set for m_external_location
      const size_t num_elements = std::reduce(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>());
      m_tensor_data_size = is_raw ? (num_elements * element_type.bitwidth() + 7) / 8 : num_elements;
    }
  }

  static ov::element::Type ConvertONNXTypeToOV(const ONNXTensorElementDataType& ort_type) noexcept {
    switch (ort_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ov::element::f32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return ov::element::f64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ov::element::f16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return ov::element::bf16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        return ov::element::f8e4m3;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        return ov::element::f8e5m2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
        return ov::element::i4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return ov::element::i8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return ov::element::i16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ov::element::i32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return ov::element::i64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
        return ov::element::u4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return ov::element::u8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return ov::element::u16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ov::element::u32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return ov::element::u64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ov::element::boolean;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return ov::element::string;
      default:
        return ov::element::dynamic;
    }
  }

  static std::vector<std::string> MakeStringVector(const Ort::ConstValue& value) {
    const size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      result.push_back(value.GetStringTensorElement(i));
    }
    return result;
  }

 private:
  void fetch_initializer_data(const Ort::ConstValueInfo& value_info) {
    Ort::ExternalInitializerInfo ext_info{nullptr};
    if (value_info.GetExternalInitializerInfo(ext_info).IsOK() && ext_info != nullptr) {
      const auto rel_path = std::filesystem::path(ext_info.GetFilePath());
      const auto offset = ext_info.GetFileOffset();
      const auto length = ext_info.GetByteSize();
      if (length > 0) {
        m_tensor_data = reinterpret_cast<uint8_t*>(static_cast<intptr_t>(offset));
        m_tensor_data_size = length;
        m_external_location = std::make_shared<std::string>(ToUtf8String(rel_path));
        return;
      }
    }

    Ort::ConstValue initializer_value{nullptr};
    if (value_info.GetInitializer(initializer_value).IsOK() && initializer_value != nullptr) {
      if (m_element_type == ov::element::string) {
        // OpenVINO requires std::vector<std::string> in m_tensor_data_any (not a pointer).
        m_tensor_data_any = MakeStringVector(initializer_value);
        m_is_raw = false;
        return;
      }

      const void* raw_data = initializer_value.GetTensorRawData();
      if (raw_data != nullptr) {
        m_tensor_data = reinterpret_cast<const uint8_t*>(raw_data);
        m_is_raw = true;
      }
    }
  }

  std::string name_{};
};

// ============================================================================
// ort_decoder_tensor (implements all required virtuals from DecoderBaseTensor)
// ============================================================================
class ort_decoder_tensor : public ov::frontend::onnx::DecoderBaseTensor {
 public:
  explicit ort_decoder_tensor(const Ort::ConstValueInfo value_info, const int64_t input_index, const int64_t output_index)
      : input_index_(input_index),
        output_index_(output_index),
        tensor_info_(value_info) {
  }

  // Constructor used by op_attr for tensor-typed attributes
  explicit ort_decoder_tensor(ort_tensor_meta_info&& tensor_info)
      : input_index_(-1),
        output_index_(-1),
        tensor_info_(std::move(tensor_info)) {
  }

  ~ort_decoder_tensor() override = default;
  OVEP_DISABLE_COPY_AND_MOVE(ort_decoder_tensor)

  const ov::frontend::onnx::TensorMetaInfo& get_tensor_info() const override { return tensor_info_; }
  int64_t get_input_idx() const override { return input_index_; }
  int64_t get_output_idx() const override { return output_index_; }
  ov::Any get_attribute(const std::string&) const override { return nullptr; }
  size_t get_input_size() const override { return 0; }                                // tensors don't have inputs
  void get_input_node(size_t, std::string&, std::string&, size_t&) const override {}  // tensors don't have inputs

  const std::string& get_op_type() const override { return empty_str_; }
  const std::string& get_op_name() const override { return empty_str_; }

 private:
  int64_t input_index_{0};
  int64_t output_index_{0};
  ort_tensor_meta_info tensor_info_;
  inline static const std::string empty_str_{};
};

// ============================================================================
// op_attr: helper to decode ORT attributes
// ============================================================================
struct op_attr {
  explicit op_attr(const Ort::ConstOpAttr ort_attr) {
    name_ = ort_attr.GetName();

    auto get = [&]<typename T>() -> T { T v{}; ort_attr.GetValue(v); return v; };
    auto get_arr = [&]<typename T>() -> std::vector<T> { std::vector<T> v; ort_attr.GetValueArray(v); return v; };

    switch (ort_attr.GetType()) {
      case ORT_OP_ATTR_INT:
        value_ = get.template operator()<int64_t>();
        break;
      case ORT_OP_ATTR_INTS:
        value_ = get_arr.template operator()<int64_t>();
        break;
      case ORT_OP_ATTR_FLOAT:
        value_ = get.template operator()<float>();
        break;
      case ORT_OP_ATTR_FLOATS:
        value_ = get_arr.template operator()<float>();
        break;
      case ORT_OP_ATTR_STRING:
        value_ = get.template operator()<std::string>();
        break;
      case ORT_OP_ATTR_STRINGS:
        value_ = get_arr.template operator()<std::string>();
        break;
      case ORT_OP_ATTR_TENSOR: {
        // Wrap as DecoderBaseTensor; tensor_attr_value_ keeps the buffer alive so m_tensor_data stays valid (zero-copy).
        if (ort_attr.GetTensorAttributeAsOrtValue(tensor_attr_value_).IsOK() && tensor_attr_value_ != nullptr) {
          ort_tensor_meta_info meta{name_};
          const Ort::TensorTypeAndShapeInfo info = tensor_attr_value_.GetTensorTypeAndShapeInfo();
          meta.m_element_type = ort_tensor_meta_info::ConvertONNXTypeToOV(info.GetElementType());
          if (meta.m_element_type == ov::element::string) {
            // OpenVINO requires std::vector<std::string> in m_tensor_data_any.
            meta.m_tensor_data_any = ort_tensor_meta_info::MakeStringVector(tensor_attr_value_.GetConst());
            meta.m_is_raw = false;
          } else {
            const void* raw = tensor_attr_value_.GetTensorRawData();
            if (raw != nullptr) {
              meta.m_tensor_data = reinterpret_cast<const uint8_t*>(raw);
              meta.m_is_raw = true;
            }
          }
          meta.apply_shape_and_size(info.GetShape(), meta.m_element_type, meta.m_is_raw);
          value_ = std::static_pointer_cast<ov::frontend::onnx::DecoderBase>(
              std::make_shared<ort_decoder_tensor>(std::move(meta)));
        }
        break;
      }
      default:
        break;
    }
  }

  std::string name_;
  ov::Any value_;
  Ort::Value tensor_attr_value_{nullptr};  // keeps ORT_OP_ATTR_TENSOR data alive (zero-copy)
};

// ============================================================================
// ort_decoder_operation
// ============================================================================
class ort_decoder_operation : public ov::frontend::onnx::DecoderBaseOperation {
 public:
  ort_decoder_operation(const Ort::ConstNode node, const uint64_t opset)
      : node_(node), opset_(opset) {
    if (node_ == nullptr) ORT_CXX_API_THROW("Expected node_ to be non-null", ORT_EP_FAIL);

    op_name_ = node_.GetName();
    default_domain_ = node_.GetDomain();
    op_type_ = node_.GetOperatorType();

    for (const auto& ort_attr : node_.GetAttributes()) {
      attributes_.emplace_back(ort_attr);
    }
    attr_index_.reserve(attributes_.size());
    for (size_t i = 0; i < attributes_.size(); ++i) {
      attr_index_.emplace(attributes_[i].name_, i);
    }

    // nullptr entries = missing optional inputs; use empty name as placeholder.
    const auto inputs = node_.GetInputs();
    input_tensor_info_.reserve(inputs.size());
    for (const Ort::ConstValueInfo& value_info : inputs) {
      if (value_info == nullptr) {
        input_tensor_info_.emplace_back("");
      } else if (value_info.IsConstantInitializer()) {
        input_tensor_info_.emplace_back(value_info, ort_tensor_meta_info::initializer_tag{});
      } else {
        input_tensor_info_.emplace_back(value_info, std::nullopt);
      }
    }

    const auto outputs = node_.GetOutputs();
    output_tensor_info_.reserve(outputs.size());
    for (const Ort::ConstValueInfo& value_info : outputs) {
      if (value_info == nullptr) {
        output_tensor_info_.emplace_back("");
      } else {
        output_tensor_info_.emplace_back(value_info, std::nullopt);
      }
    }
  }

  const std::string& get_op_type() const override { return op_type_; }
  const std::string& get_op_name() const override { return op_name_; }

  void get_input_node(size_t, std::string&, std::string&, size_t&) const override final {}  // not applicable for op decoders

  ov::Any get_attribute(const std::string& attr) const override final {
    auto sg_it = subgraph_attrs_.find(attr);
    if (sg_it != subgraph_attrs_.end()) {
      return sg_it->second;
    }
    auto it = attr_index_.find(attr);
    if (it != attr_index_.end()) {
      return attributes_[it->second].value_;
    }
    return ov::Any{};
  }

  size_t get_input_size() const override { return input_tensor_info_.size(); }
  size_t get_output_size() const override { return output_tensor_info_.size(); }

  const std::string& get_input_tensor_name(size_t idx) const override {
    return idx >= input_tensor_info_.size() ? error_str_ : input_tensor_info_.at(idx).name();
  }
  const std::string& get_output_tensor_name(size_t idx) const override {
    return idx >= output_tensor_info_.size() ? error_str_ : output_tensor_info_.at(idx).name();
  }

  ov::element::Type get_input_tensor_type(size_t idx) const override {
    return idx >= input_tensor_info_.size() ? ov::element::dynamic : input_tensor_info_.at(idx).m_element_type;
  }
  ov::element::Type get_output_tensor_type(size_t idx) const override {
    return idx >= output_tensor_info_.size() ? ov::element::dynamic : output_tensor_info_.at(idx).m_element_type;
  }

  const ov::frontend::onnx::TensorMetaInfo& get_input_tensor_info(size_t idx) const override {
    return idx >= input_tensor_info_.size() ? error_tensor_meta_ : input_tensor_info_.at(idx);
  }
  const ov::frontend::onnx::TensorMetaInfo& get_output_tensor_info(size_t idx) const override {
    return idx >= output_tensor_info_.size() ? error_tensor_meta_ : output_tensor_info_.at(idx);
  }

  const std::string& get_domain() const override { return default_domain_; }
  uint64_t get_op_set() const override { return opset_; }
  bool has_attribute(const std::string& name) const override {
    return attr_index_.count(name) != 0 || subgraph_attrs_.count(name) != 0;
  }
  void experimental_get_internal_structures(const void** node_def) const override { *node_def = static_cast<const OrtNode*>(node_); }
  const std::string& get_name() const override { return op_name_; }

  void set_subgraph_attrs(std::map<std::string, std::shared_ptr<ov::frontend::onnx::GraphIterator>> attrs) {
    subgraph_attrs_ = std::move(attrs);
  }

 private:
  Ort::ConstNode node_{nullptr};
  uint64_t opset_{0};
  std::string op_name_;
  std::string op_type_;
  std::string default_domain_{"ai.onnx"};
  std::vector<ort_tensor_meta_info> input_tensor_info_;
  std::vector<ort_tensor_meta_info> output_tensor_info_;
  std::vector<op_attr> attributes_;
  std::unordered_map<std::string, size_t> attr_index_;
  std::map<std::string, std::shared_ptr<ov::frontend::onnx::GraphIterator>> subgraph_attrs_;
  inline static const std::string error_str_{"error"};
  inline static ort_tensor_meta_info error_tensor_meta_;
};

// ============================================================================
// ort_graph_delegate — lazy decoder construction
//
// The OV ONNX frontend iterates via reset()/next()/get_decoder() exactly once.
// We store only lightweight ORT handles (Ort::ConstValueInfo / Ort::ConstNode)
// and section boundaries, then build a decoder on-demand per get_decoder() call.
//
// Initializers are NOT emitted as separate decoder items. The OV frontend's
// load_model() discards all initializer-type decoders that have data
// (has_data && output_idx < 0 -> continue). Instead, initializer data is
// discovered when op input tensors are decoded: ort_tensor_meta_info calls
// IsConstantInitializer() + GetInitializer() which provides the data pointer.
//
// Logical layout (flat index):
//   [0 .. end_inputs)           — graph inputs
//   [end_inputs .. end_outputs) — graph outputs
//   [end_outputs .. total_size) — operation nodes
// ============================================================================
class ort_graph_delegate : public ov::frontend::onnx::GraphIterator {
 public:
  ort_graph_delegate(const Ort::ConstGraph graph, std::filesystem::path model_dir, Ort::Logger logger)
      : graph_(graph), model_dir_(std::move(model_dir)), logger_(logger) {
    if (graph_ == nullptr) ORT_CXX_API_THROW("Expected graph_ to be non-null", ORT_EP_FAIL);

    opset_ = static_cast<uint64_t>(graph_.GetOnnxIRVersion());
    graph_inputs_ = graph_.GetInputs();
    graph_outputs_ = graph_.GetOutputs();
    nodes_ = graph_.GetNodes();

    end_inputs_ = graph_inputs_.size();
    end_outputs_ = end_inputs_ + graph_outputs_.size();
    total_size_ = end_outputs_ + nodes_.size();
  }

  size_t size() const override { return total_size_; }
  void reset() override { index_ = 0; }
  void next() override {
    if (index_ < total_size_) ++index_;
  }
  bool is_end() const override { return index_ >= total_size_; }

  std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override {
    if (index_ >= total_size_) return nullptr;

    if (index_ < end_inputs_) {
      const size_t input_idx = index_;
      return std::make_shared<ort_decoder_tensor>(
          graph_inputs_[input_idx], static_cast<int64_t>(input_idx), -1);
    }

    if (index_ < end_outputs_) {
      const size_t out_idx = index_ - end_inputs_;
      return std::make_shared<ort_decoder_tensor>(
          graph_outputs_[out_idx], -1, static_cast<int64_t>(out_idx));
    }

    const size_t node_idx = index_ - end_outputs_;
    const auto& graph_node = nodes_[node_idx];
    auto op_decoder = std::make_shared<ort_decoder_operation>(graph_node, graph_node.GetSinceVersion());

    // Subgraph-bearing ops (If/Loop/Scan) expose their bodies as attributes; wrap
    // each as a nested delegate so the OV frontend can recurse into it.
    std::map<std::string, std::shared_ptr<ov::frontend::onnx::GraphIterator>> subgraph_attrs;
    for (const auto& [attr_name, subgraph] : graph_node.GetSubgraphs()) {
      subgraph_attrs[attr_name] = std::make_shared<ort_graph_delegate>(Ort::ConstGraph{subgraph}, model_dir_, logger_);
    }
    if (!subgraph_attrs.empty()) {
      op_decoder->set_subgraph_attrs(std::move(subgraph_attrs));
    }
    return op_decoder;
  }

  int64_t get_opset_version(const std::string&) const override { return static_cast<int64_t>(opset_); }
  std::filesystem::path get_model_dir() const override { return model_dir_; }

  std::map<std::string, std::string> get_metadata() const override {
    std::map<std::string, std::string> result;
    try {
      Ort::ModelMetadata metadata = graph_.GetModelMetadata();
      Ort::AllocatorWithDefaultOptions allocator;

      auto add_string = [&](Ort::AllocatedStringPtr ptr, const char* key) {
        if (ptr) result[key] = ptr.get();
      };

      add_string(metadata.GetDescriptionAllocated(allocator), "doc_string");
      add_string(metadata.GetDomainAllocated(allocator), "domain");
      add_string(metadata.GetProducerNameAllocated(allocator), "producer_name");

      result["model_version"] = std::to_string(metadata.GetVersion());

      for (const auto& key_ptr : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
        if (key_ptr) {
          auto val = metadata.LookupCustomMetadataMapAllocated(key_ptr.get(), allocator);
          if (val) result[key_ptr.get()] = val.get();
        }
      }
    } catch (const std::exception& e) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, e.what());
    } catch (...) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, "Unknown error reading model metadata");
    }
    return result;
  }

 private:
  Ort::ConstGraph graph_{nullptr};
  std::filesystem::path model_dir_;
  Ort::Logger logger_;
  uint64_t opset_{0};

  // Raw ORT handles — no decoder objects allocated until get_decoder().
  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstNode> nodes_;

  // Section boundaries for flat index.
  size_t end_inputs_{0};
  size_t end_outputs_{0};
  size_t total_size_{0};
  size_t index_{0};
};

inline std::shared_ptr<ov::frontend::onnx::GraphIterator> get_delegate_instance(const OrtGraph* model, std::filesystem::path model_dir, Ort::Logger logger) {
  return std::make_shared<ort_graph_delegate>(Ort::ConstGraph{model}, std::move(model_dir), logger);
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
