// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime {

///////////////////////////////////////////////////////////////////////
// Concept describing any container with .data() and .size() members
template <typename T>
concept contiguous_sequence = requires(T& t, const T& ct) {
  // Must have a data() member that returns a pointer
  {
    ct.data()
  } -> std::convertible_to<std::add_pointer_t<const typename std::remove_reference_t<decltype(*ct.data())> > >;
  // Must have a size() member that returns an integral type
  { ct.size() } -> std::convertible_to<std::size_t>;
};

///////////////////////////////////////////////////////////////////////
// Helper to extract element type from a contiguous_sequence container
template <contiguous_sequence T>
using element_type_t = std::remove_reference_t<decltype(*std::declval<T>().data())>;

///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct pipeline_wrapper : common::pipeline_t {
  using device_t = typename fixture_t::device_wrapper_t;
  using graph_t = typename fixture_t::graph_wrapper_t;
  using session_t = typename fixture_t::session_wrapper_t;
  using memory_info_t = typename fixture_t::memory_info_wrapper_t;
  using io_binding_t = typename fixture_t::io_binding_wrapper_t;

  using byte_data_t = std::vector<std::byte>;

  template <typename data_type_t, contiguous_sequence container_t = std::vector<data_type_t> >
  using io_data = std::map<std::string, container_t>;

  device_t& device_;
  graph_t& graph_;

  memory_info_t memory_info_;
  Ort::Allocator allocator_;

  io_binding_t io_binding_;

  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<Ort::Value> input_tensors_;
  std::vector<Ort::Value> output_tensors_;

  ///////////////////////////////////////////////////////////////////////
  pipeline_wrapper(graph_t& graph) : pipeline_wrapper(graph.device_, graph) {}

  ///////////////////////////////////////////////////////////////////////
  pipeline_wrapper(device_t& device, graph_t& graph)
      : device_(device), graph_(graph), memory_info_(device, OrtDeviceMemoryType_HOST_ACCESSIBLE), allocator_(*graph_.session_, memory_info_), io_binding_(*graph_.session_) {
    setup_io();
  }

  ///////////////////////////////////////////////////////////////////////
  // Allocate tensors from a caller-provided memory info (e.g. a plain CPU
  // allocator). The caller must keep mem_info alive for this pipeline's lifetime.
  pipeline_wrapper(device_t& device, graph_t& graph, Ort::ConstMemoryInfo mem_info)
      : device_(device), graph_(graph), memory_info_(mem_info), allocator_(*graph_.session_, memory_info_), io_binding_(*graph_.session_) {
    setup_io();
  }

  ///////////////////////////////////////////////////////////////////////
  pipeline_wrapper(
      device_t& device,
      graph_t& graph,
      const std::vector<int64_t>& input_shape,
      const std::vector<int64_t>& output_shape)
      : device_(device), graph_(graph), memory_info_(device, OrtDeviceMemoryType_HOST_ACCESSIBLE), allocator_(*graph_.session_, memory_info_), io_binding_(*graph_.session_) {
    for (auto& [name, desc] : graph_.input_desc_map_) {
      input_names_.emplace_back(name.c_str());
      input_tensors_.emplace_back(
          Ort::Value::CreateTensor(allocator_, input_shape.data(), input_shape.size(), desc.element_type_));

      if (graph_.network_.ort_enable_io_binding_) {
        io_binding_.bind_input(name, input_tensors_.back());
      }
    }

    for (auto& [name, desc] : graph_.output_desc_map_) {
      output_names_.emplace_back(name.c_str());
      output_tensors_.emplace_back(
          Ort::Value::CreateTensor(allocator_, output_shape.data(), output_shape.size(), desc.element_type_));

      if (graph_.network_.ort_enable_io_binding_) {
        io_binding_.bind_output(name, output_tensors_.back());
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  void run(void) {
    if (graph_.network_.ort_enable_io_binding_) {
      graph_.run(io_binding_);
    } else {
      graph_.run(input_names_, input_tensors_, output_names_, output_tensors_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  void start_async(void) { OVEP_TESTS_THROW("Not Implemented!"); }

  ///////////////////////////////////////////////////////////////////////////
  void wait_for_completion(void) { OVEP_TESTS_THROW("Not Implemented!"); }

  ///////////////////////////////////////////////////////////////////////
  template <contiguous_sequence container_t>
  void upload_input_data(const std::map<std::string, container_t>& input_data) {
    using data_type_t = element_type_t<container_t>;
    for (size_t idx = 0; idx < input_names_.size(); ++idx) {
      auto input_tensor_ptr = input_tensors_[idx].GetTensorMutableRawData();
      const auto& input = input_data.at(input_names_[idx]);
      auto size_in_bytes = input.size() * sizeof(data_type_t);

      std::copy_n(
          reinterpret_cast<const char*>(input.data()),
          size_in_bytes,
          reinterpret_cast<char*>(input_tensor_ptr));
    }
  }

  ///////////////////////////////////////////////////////////////////////
  template <contiguous_sequence container_t>
  void readback_output_data(std::map<std::string, container_t>& output_data) {
    using data_type_t = element_type_t<container_t>;
    for (size_t idx = 0; idx < output_names_.size(); ++idx) {
      auto output_tensor_ptr = output_tensors_[idx].GetTensorRawData();
      auto& output = output_data.at(output_names_[idx]);
      auto size_in_bytes = output.size() * sizeof(data_type_t);

      std::copy_n(
          reinterpret_cast<const char*>(output_tensor_ptr),
          size_in_bytes,
          reinterpret_cast<char*>(output.data()));
    }
  }

  ///////////////////////////////////////////////////////////////////////
  template <contiguous_sequence input_container_t, contiguous_sequence output_container_t>
  void run(
      const std::map<std::string, input_container_t>& input_data,
      std::map<std::string, output_container_t>& output_data) {
    upload_input_data(input_data);
    run();
    readback_output_data(output_data);
  }

 private:
  ///////////////////////////////////////////////////////////////////////
  // Allocate input/output tensors from allocator_ and bind them when the
  // network requests io_binding. Shared by the static-shape constructors.
  void setup_io() {
    for (auto& [name, desc] : graph_.input_desc_map_) {
      if (!desc.is_partial_shape_) {
        input_names_.emplace_back(name.c_str());
        input_tensors_.emplace_back(Ort::Value::CreateTensor(
            allocator_,
            desc.shape_.data(),
            desc.shape_.size(),
            desc.element_type_));

        if (graph_.network_.ort_enable_io_binding_) {
          io_binding_.bind_input(name, input_tensors_.back());
        }
      } else {
        OVEP_TESTS_THROW(
            "Input (%s) has partial shape, use reshape_input or free dim override to set a static shape",
            name.c_str());
      }
    }

    for (auto& [name, desc] : graph_.output_desc_map_) {
      if (!desc.is_partial_shape_) {
        output_names_.emplace_back(name.c_str());
        output_tensors_.emplace_back(Ort::Value::CreateTensor(
            allocator_,
            desc.shape_.data(),
            desc.shape_.size(),
            desc.element_type_));

        if (graph_.network_.ort_enable_io_binding_) {
          io_binding_.bind_output(name, output_tensors_.back());
        }
      } else {
        if (graph_.network_.ort_enable_io_binding_) {
          io_binding_.bind_output(name, memory_info_);
        } else {
          OVEP_TESTS_THROW(
              "Output (%s) has partial shape, requires io_binding", name.c_str());
        }
      }
    }
  }
};
}  // namespace onnxruntime
}  // namespace ovep_tests
