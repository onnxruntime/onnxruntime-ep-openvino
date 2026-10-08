// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime {
///////////////////////////////////////////////////////////////////////
// Common base for input and output argument descriptors
template <typename fixture_t>
struct graph_arg_desc_base {
  using session_t = typename fixture_t::session_wrapper_t;

  std::string name_{};
  Ort::TypeInfo type_info_{};
  Ort::ConstTensorTypeAndShapeInfo shape_info_{};
  std::vector<int64_t> shape_{};
  ONNXTensorElementDataType element_type_{};
  bool is_partial_shape_ = false;
  size_t size_in_bytes_ = 0;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(graph_arg_desc_base);

  ///////////////////////////////////////////////////////////////////////
  graph_arg_desc_base(const session_t& session, size_t index, bool is_input)
      : name_(session.get_tensor_name(index, is_input)), type_info_(session.get_type_info(index, is_input)), shape_info_(type_info_.GetTensorTypeAndShapeInfo()), shape_(shape_info_.GetShape()), element_type_(shape_info_.GetElementType()), is_partial_shape_(std::find(shape_.begin(), shape_.end(), -1) != shape_.end()), size_in_bytes_(is_partial_shape_ ? 0 : session.get_element_size_in_bytes(element_type_) * shape_info_.GetElementCount()) {}
};

///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct graph_input_arg_desc : public graph_arg_desc_base<fixture_t> {
  using base_t = graph_arg_desc_base<fixture_t>;
  using session_t = typename base_t::session_t;

  ///////////////////////////////////////////////////////////////////////
  graph_input_arg_desc(const session_t& session, size_t index) : base_t(session, index, true) {}
};

///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct graph_output_arg_desc : public graph_arg_desc_base<fixture_t> {
  using base_t = graph_arg_desc_base<fixture_t>;
  using session_t = typename base_t::session_t;

  ///////////////////////////////////////////////////////////////////////
  graph_output_arg_desc(const session_t& session, size_t index) : base_t(session, index, false) {}
};

///////////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct graph_wrapper {
  using device_t = typename fixture_t::device_wrapper_t;
  using input_arg_desc_t = graph_input_arg_desc<fixture_t>;
  using output_arg_desc_t = graph_output_arg_desc<fixture_t>;
  using input_desc_map_t = std::map<std::string, input_arg_desc_t>;
  using output_desc_map_t = std::map<std::string, output_arg_desc_t>;
  using session_options_t = typename fixture_t::session_options_wrapper_t;
  using session_factory_t = typename fixture_t::session_factory_t;
  using session_t = typename fixture_t::session_wrapper_t;
  using io_binding_t = typename fixture_t::io_binding_wrapper_t;

  const network_t& network_;

  input_desc_map_t input_desc_map_;
  output_desc_map_t output_desc_map_;

  session_options_t session_options_;

  std::unique_ptr<session_t> session_;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(graph_wrapper);

  ///////////////////////////////////////////////////////////////////////////
  graph_wrapper(const network_t& network, session_options_t session_options)
      : network_(network), session_options_(std::move(session_options)), session_(session_factory_t::create(network, session_options_)) {
    for (size_t index = 0; index < session_->input_count_; ++index) {
      input_desc_map_.try_emplace(session_->get_tensor_name(index, true), *session_, index);
    }

    for (size_t index = 0; index < session_->output_count_; ++index) {
      output_desc_map_.try_emplace(session_->get_tensor_name(index, false), *session_, index);
    }

    if (!network_.ort_ep_dynamic_options_.empty()) {
      session_->set_ep_dynamic_options(network_.ort_ep_dynamic_options_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  graph_wrapper(device_t& device, const network_t& network)
      : network_(network), session_options_(device, network), session_(session_factory_t::create(network, session_options_)) {
    for (size_t index = 0; index < session_->input_count_; ++index) {
      input_desc_map_.try_emplace(session_->get_tensor_name(index, true), *session_, index);
    }

    for (size_t index = 0; index < session_->output_count_; ++index) {
      output_desc_map_.try_emplace(session_->get_tensor_name(index, false), *session_, index);
    }

    if (!network_.ort_ep_dynamic_options_.empty()) {
      session_->set_ep_dynamic_options(network_.ort_ep_dynamic_options_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  void run(
      const std::vector<const char*>& input_names,
      const std::vector<Ort::Value>& input_tensors,
      const std::vector<const char*>& output_names,
      std::vector<Ort::Value>& output_tensors) {
    session_->run(input_names, input_tensors, output_names, output_tensors);
  }

  ///////////////////////////////////////////////////////////////////////////
  void run(const io_binding_t& io_binding) { session_->run(io_binding); }

  ///////////////////////////////////////////////////////////////////////
  inline auto input_count(void) const { return input_desc_map_.size(); }
  inline auto output_count(void) const { return output_desc_map_.size(); }
  inline auto input_name(void) const { return input_desc_map_.begin()->first; }
  inline auto output_name(void) const { return output_desc_map_.begin()->first; }
  inline auto input_names(void) const {
    std::vector<const char*> names;
    for (const auto& [name, _] : input_desc_map_) {
      names.push_back(name.c_str());
    }

    return names;
  }
  inline auto output_names(void) const {
    std::vector<const char*> names;
    for (const auto& [name, _] : output_desc_map_) {
      names.push_back(name.c_str());
    }

    return names;
  }
  inline auto find_input(const std::string& name) const {
    return input_desc_map_.find(name) != input_desc_map_.end();
  }
  inline auto find_output(const std::string& name) const {
    return output_desc_map_.find(name) != output_desc_map_.end();
  }
  inline auto input_size(const std::string& name) const {
    return input_desc_map_.find(name)->second.size_in_bytes_;
  }
  inline auto output_size(const std::string& name) const {
    return output_desc_map_.find(name)->second.size_in_bytes_;
  }
  inline auto input_precision(const std::string& name) const {
    return to_string(input_desc_map_.find(name)->second.element_type_);
  }
  inline auto output_precision(const std::string& name) const {
    return to_string(output_desc_map_.find(name)->second.element_type_);
  }
  inline auto input_layout(const std::string& name) const { return "NCHW"; }
  inline auto output_layout(const std::string& name) const { return "NCHW"; }
};
}  // namespace onnxruntime
}  // namespace ovep_tests
