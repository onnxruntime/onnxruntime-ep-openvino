// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// Ensure compile time ORT API version is at least 30
static_assert(ORT_API_VERSION >= 30, "ORT API Version 30 or higher is required to build OpenVINO EP");

namespace onnxruntime {
namespace openvino_ep_plugin {

inline uint32_t ort_runtime_api_version = ORT_API_VERSION;
inline constexpr uint32_t ort_compile_time_api_version = ORT_API_VERSION;
inline constexpr uint32_t ort_min_api_version_supported = 23;  // Plugin EP minimum supported ORT API version

// True when the ORT headers we compiled against are at least MinVersion.
template <uint32_t MinVersion>
concept OrtApiAtLeast = (ORT_API_VERSION >= MinVersion);

// Gates a call on compile-time (concept) and runtime ORT version. Pass the base ORT
// struct pointer first so the lambda receives it as auto*, making versioned field
// accesses template-dependent and safe to compile against older headers.
template <uint32_t MinVersion, typename EpT, typename Func>
  requires OrtApiAtLeast<MinVersion>
void OrtVersionGate(EpT* ep, Func&& func) {
  if (ort_runtime_api_version >= MinVersion)
    std::forward<Func>(func)(ep);
}

template <uint32_t MinVersion, typename EpT, typename Func>
void OrtVersionGate(EpT* /*ep*/, Func&& /*func*/) {}

// Overload without a struct pointer — for versioned behaviour that does not
// access new struct fields and needs no template-dependence trick.
template <uint32_t MinVersion, typename Func>
  requires OrtApiAtLeast<MinVersion>
void OrtVersionGate(Func&& func) {
  if (ort_runtime_api_version >= MinVersion)
    std::forward<Func>(func)();
}

template <uint32_t MinVersion, typename Func>
void OrtVersionGate(Func&& /*func*/) {}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
