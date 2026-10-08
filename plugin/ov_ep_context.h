// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <format>
#include <gsl/narrow>

#include "ov_provider.h"
#include "ov_common_utils.h"
#include "ov_shared_context.h"
#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

struct EpContextNode {
  int64_t main_context{1};
  int64_t embed_mode{1};
  std::optional<int64_t> max_size{0};
  std::string ep_cache_context;
  std::string ep_sdk_version;
  std::string onnx_model_filename;
  std::string hardware_architecture;
  std::string partition_name;
  std::string source;
  std::string notes;

  enum class EpContextType {
    Native,
    OV_IR,
  };

  struct private_fields {
    EpContextType type{EpContextType::Native};
    std::string node_name{"OpenVINO_EP_Node"};
    std::filesystem::path epctx_dir{};
    std::shared_ptr<SharedContext> shared_context{nullptr};
  } private_fields_;

  static constexpr const char* ep_context_op_type = "EPContext";
  static constexpr const char* ep_context_domain = "com.microsoft";

  // Default constructor for manual initialization
  explicit EpContextNode() = default;

  // Constructs an EpContextNode that initializes its fields from an OrtNode
  EpContextNode(const OrtNode* node, const std::filesystem::path& ep_context_path, const std::filesystem::path& model_path) {
    Ort::ConstNode ort_node{node};

    // Helper lambda to extract int64 attribute values
    auto get_attr_int64 = [&](const char* name, int64_t default_val) -> int64_t {
      Ort::ConstOpAttr attr{nullptr};
      auto status = ort_node.GetAttributeByName(name, attr);
      if (!status.IsOK() || attr == nullptr || attr.GetType() != ORT_OP_ATTR_INT) {
        return default_val;
      }
      int64_t val = 0;
      Ort::ThrowOnError(attr.GetValue(val));
      return val;
    };

    // Helper lambda to extract string attribute values
    auto get_attr_string = [&](const char* name) -> std::string {
      Ort::ConstOpAttr attr{nullptr};
      auto status = ort_node.GetAttributeByName(name, attr);
      if (!status.IsOK() || attr == nullptr || attr.GetType() != ORT_OP_ATTR_STRING) {
        return std::string{};
      }
      std::string val;
      Ort::ThrowOnError(attr.GetValue(val));
      return val;
    };

    main_context = get_attr_int64("main_context", 1);
    ep_cache_context = get_attr_string("ep_cache_context");
    embed_mode = get_attr_int64("embed_mode", 1);
    ep_sdk_version = get_attr_string("ep_sdk_version");
    onnx_model_filename = get_attr_string("onnx_model_filename");
    hardware_architecture = get_attr_string("hardware_architecture");
    partition_name = get_attr_string("partition_name");
    source = get_attr_string("source");
    notes = get_attr_string("notes");
    max_size = get_attr_int64("max_size", 0);

    private_fields_.epctx_dir = ep_context_path.parent_path();

    if (embed_mode == 1) {
      if (openvino_ep::utils::IsXmlHeader(ep_cache_context)) {
        private_fields_.type = EpContextType::OV_IR;
      }
    } else {
      if (ep_cache_context.ends_with(".xml")) {
        private_fields_.epctx_dir = model_path.parent_path();
      }
      auto ep_ctx_path = GetEpContextPath();

      // External blob may not be on disk when the host delegates I/O via the v28 EPContext data
      // read callback. In that case keep the default Native type; the actual bytes are validated
      // when the buffer is deserialized later.
      std::ifstream ep_ctx_file(ep_ctx_path, std::ios::binary);
      if (!ep_ctx_file.fail()) {
        if (openvino_ep::utils::IsModelStreamXML(ep_ctx_file)) {
          private_fields_.type = EpContextType::OV_IR;
        }
      }
    }
  }

  static bool IsEpContextNode(const Ort::ConstNode& node) {
    return node.GetOperatorType() == ep_context_op_type &&
           node.GetDomain() == ep_context_domain;
  }

  std::filesystem::path GetEpContextPath() const {
    if (embed_mode == 1) {
      return std::filesystem::path{};
    }
    const auto relative_location = PathFromUtf8(ep_cache_context);
    OVEP_ENFORCE(IsSafeRelativeSubpath(relative_location),
                 "Error: ep_cache_context must be a relative path within the model directory: ", ep_cache_context);
    return private_fields_.epctx_dir / relative_location;
  }

  // Create an Ort::Node representing this EpContextNode with inputs/outputs from io_map
  Ort::Node CreateNode(const OnnxIOMapping& io_map) const {
    // Create attributes using C++ API
    std::vector<Ort::OpAttr> attributes;
    auto add_attr_string = [&](const char* name, auto& value) {
      if (!value.empty()) {
        attributes.emplace_back(name, value.c_str(), gsl::narrow<int>(value.size()), ORT_OP_ATTR_STRING);
      }
    };
    auto add_attr_int64 = [&](const char* name, int64_t value) {
      attributes.emplace_back(name, &value, 1, ORT_OP_ATTR_INT);
    };

    add_attr_int64("main_context", main_context);
    add_attr_int64("embed_mode", embed_mode);
    if (max_size.has_value()) {
      add_attr_int64("max_size", max_size.value());
    }
    add_attr_string("ep_cache_context", ep_cache_context);
    add_attr_string("ep_sdk_version", ep_sdk_version);
    add_attr_string("onnx_model_filename", onnx_model_filename);
    add_attr_string("hardware_architecture", hardware_architecture);
    add_attr_string("partition_name", partition_name);
    add_attr_string("source", source);
    add_attr_string("notes", notes);

    return Ort::Node(
        ep_context_op_type,
        ep_context_domain,
        private_fields_.node_name,
        io_map.input_names,
        io_map.output_names,
        attributes);
  }
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
