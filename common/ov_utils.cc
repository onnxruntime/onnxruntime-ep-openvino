// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

std::string GetEnvironmentVar(std::string_view env_var) {
#ifdef _WIN32
  DWORD size = GetEnvironmentVariableA(env_var.data(), nullptr, 0);
  if (size == 0) return std::string{};
  std::string value(size - 1, '\0');
  GetEnvironmentVariableA(env_var.data(), value.data(), size);
  return value;
#else
  const char* value = std::getenv(env_var.data());
  return value ? std::string(value) : std::string{};
#endif
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
