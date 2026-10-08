// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime {
///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct io_binding_wrapper : Ort::IoBinding {
  using session_t = typename fixture_t::session_wrapper_t;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(io_binding_wrapper);

  ///////////////////////////////////////////////////////////////////////
  io_binding_wrapper(session_t& session) : Ort::IoBinding(session) {}

  ///////////////////////////////////////////////////////////////////////
  void bind_input(const std::string& name, const Ort::Value& value) {
    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindInput( io_binding: " << this << ", name: " << name
                         << ", value: " << value << " ): {";

    BindInput(name.c_str(), value);

    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindInput: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  void bind_output(const std::string& name, const Ort::Value& value) {
    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindOutput( io_binding: " << this << ", name: " << name
                         << ", value: " << value << " ): {";

    BindOutput(name.c_str(), value);

    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindOutput: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  void bind_output(const std::string& name, const Ort::ConstMemoryInfo& memory_info) {
    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindOutput( io_binding: " << this << ", name: " << name
                         << ", memory_info: " << memory_info << " ): {";

    BindOutput(name.c_str(), memory_info);

    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::BindOutput: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  void synchronize_inputs(void) {
    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::SynchronizeInputs( io_binding: " << this << " ): {";

    SynchronizeInputs();

    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::SynchronizeInputs: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  void synchronize_outputs(void) {
    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::SynchronizeOutputs( io_binding: " << this << " ): {";

    SynchronizeOutputs();

    OVEP_TESTS_LOG_DEBUG << "OrtIoBinding::SynchronizeOutputs: ...OK }";
  }
};

}  // namespace onnxruntime
}  // namespace ovep_tests
