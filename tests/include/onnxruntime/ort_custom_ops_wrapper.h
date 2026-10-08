// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <onnxruntime/onnxruntime_lite_custom_op.h>

namespace ovep_tests {
namespace onnxruntime {

///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct custom_ops_wrapper {
  using ort_custom_op_t = Ort::Custom::OrtLiteCustomOp;

  std::vector<std::unique_ptr<ort_custom_op_t> > custom_ops_;
  Ort::CustomOpDomain custom_op_domain_{"com.intel.etests"};

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(custom_ops_wrapper);

  ///////////////////////////////////////////////////////////////////////
  custom_ops_wrapper(Ort::SessionOptions& session_options) {
    custom_ops_.emplace_back(Ort::Custom::CreateLiteCustomOp(
        "Identity",
        "CPUExecutionProvider",
        identity_compute_f16,
        identity_shape_inference));

    custom_ops_.emplace_back(Ort::Custom::CreateLiteCustomOp(
        "IdentityFloat",
        "CPUExecutionProvider",
        identity_compute_f32,
        identity_shape_inference));

    for (const auto& custom_op : custom_ops_) {
      custom_op_domain_.Add(custom_op.get());
    }
    session_options.Add(custom_op_domain_);
  }

 private:
  ///////////////////////////////////////////////////////////////////////
  static Ort::Status identity_compute_f16(
      const Ort::Custom::Tensor<Ort::Float16_t>& X,
      Ort::Custom::Tensor<Ort::Float16_t>& Z) {
    auto input_shape = X.Shape();
    auto x_raw = X.Data();
    auto z_raw = Z.Allocate(input_shape);
    std::memcpy(z_raw, x_raw, X.SizeInBytes());
    return {};
  }

  ///////////////////////////////////////////////////////////////////////
  static Ort::Status identity_compute_f32(
      const Ort::Custom::Tensor<float>& X,
      Ort::Custom::Tensor<float>& Z) {
    auto input_shape = X.Shape();
    auto x_raw = X.Data();
    auto z_raw = Z.Allocate(input_shape);
    std::memcpy(z_raw, x_raw, X.SizeInBytes());
    return {};
  }

  static Ort::Status identity_shape_inference(Ort::ShapeInferContext& shape_ctx) {
    shape_ctx.SetOutputShape(0, shape_ctx.GetInputShape(0));
    return {};
  }
};

}  // namespace onnxruntime
}  // namespace ovep_tests
