// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

///////////////////////////////////////////////////////////////////////////
enum class ort_execution_provider_t {
  cpu,
  ov
};

///////////////////////////////////////////////////////////////////////////
inline std::string to_string(OrtExecutionProviderDevicePolicy policy) {
  // clang-format off
    switch( policy )
    {
        case OrtExecutionProviderDevicePolicy_DEFAULT: return "DEFAULT";
        case OrtExecutionProviderDevicePolicy_PREFER_CPU: return "PREFER_CPU";
        case OrtExecutionProviderDevicePolicy_PREFER_GPU: return "PREFER_GPU";
        case OrtExecutionProviderDevicePolicy_PREFER_NPU: return "PREFER_NPU";
        case OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY: return "MAX_EFFICIENCY";
        case OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE: return "MAX_PERFORMANCE";
        case OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER: return "MIN_OVERALL_POWER";
        default: OVEP_TESTS_THROW( "Unsupported EP device policy" );
    }
  // clang-format on
}

///////////////////////////////////////////////////////////////////////////
inline const std::unordered_map<std::string, OrtExecutionProviderDevicePolicy> ort_device_policy_str_to_enum{
    {"DEFAULT", OrtExecutionProviderDevicePolicy_DEFAULT},
    {"PREFER_CPU", OrtExecutionProviderDevicePolicy_PREFER_CPU},
    {"PREFER_GPU", OrtExecutionProviderDevicePolicy_PREFER_GPU},
    {"PREFER_NPU", OrtExecutionProviderDevicePolicy_PREFER_NPU},
    {"MAX_EFFICIENCY", OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY},
    {"MAX_PERFORMANCE", OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE}};

///////////////////////////////////////////////////////////////////////////
inline OrtExecutionProviderDevicePolicy to_ort_execution_provider_device_policy(const std::string& policy) {
  auto it = ort_device_policy_str_to_enum.find(policy);
  if (it != ort_device_policy_str_to_enum.end()) {
    return it->second;
  } else {
    OVEP_TESTS_THROW("Unsupported policy for OrtExecutionProviderDevicePolicy");
  }
}

///////////////////////////////////////////////////////////////////////////
inline std::string to_string(OrtHardwareDeviceType type) {
  // clang-format off
    switch( type )
    {
        case OrtHardwareDeviceType_CPU: return "CPU";
        case OrtHardwareDeviceType_GPU: return "GPU";
        case OrtHardwareDeviceType_NPU: return "NPU";
        default: OVEP_TESTS_THROW( "Unsupported hardware device type" );
    }
  // clang-format on
}

///////////////////////////////////////////////////////////////////////////
inline std::string to_string(ExecutionMode mode) {
  // clang-format off
    switch( mode )
    {
        case ORT_SEQUENTIAL: return "sequential";
        case ORT_PARALLEL:   return "parallel";
        default: OVEP_TESTS_THROW( "Unsupported execution mode" );
    }
  // clang-format on
}

///////////////////////////////////////////////////////////////////////////
inline std::string to_string(GraphOptimizationLevel level) {
  // clang-format off
    switch( level )
    {
        case ORT_DISABLE_ALL: return "disable_all";
        case ORT_ENABLE_BASIC: return "enable_basic";
        case ORT_ENABLE_EXTENDED: return "enable_extended";
        case ORT_ENABLE_ALL: return "enable_all";
        default: OVEP_TESTS_THROW( "Unsupported execution mode" );
    }
  // clang-format on
}

///////////////////////////////////////////////////////////////////////////
inline std::string to_string(ONNXTensorElementDataType type) {
  // clang-format off
    switch( type )
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "U8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "I8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "U16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return "I16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "FP16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return "U32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "I32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "FP32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return "U64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "I64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "FP64";
        default: OVEP_TESTS_THROW( "Unsupported element data type" );
    }
  // clang-format on
}

///////////////////////////////////////////////////////////////////////
enum class embed_mode_t {
  on,
  off
};

///////////////////////////////////////////////////////////////////////
inline std::string to_string(embed_mode_t mode) {
  return mode == embed_mode_t::on ? "1" : "0";
}

///////////////////////////////////////////////////////////////////////
enum class session_model_load_mode_t {
  file,
  memory
};

///////////////////////////////////////////////////////////////////////
inline std::string to_string(session_model_load_mode_t mode) {
  return mode == session_model_load_mode_t::file ? "file" : "memory";
}

///////////////////////////////////////////////////////////////////////
enum class weight_sharing_t {
  on,
  off
};

///////////////////////////////////////////////////////////////////////
enum class ep_type_t {
  legacy,
  abi,
};

inline std::string to_string(ep_type_t ep) {
  return ep == ep_type_t::legacy ? "legacy" : "abi";
}

///////////////////////////////////////////////////////////////////////
inline std::string to_string(weight_sharing_t mode) {
  return mode == weight_sharing_t::on ? "1" : "0";
}
