// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <string>
#include <istream>
#include <utility>

namespace onnxruntime {
namespace openvino_ep {
namespace utils {

bool IsModelStreamXML(std::istream& model_stream);
static inline bool IsXmlHeader(std::string_view header) {
  return header.rfind("<?xml", 0) == 0 && header.find("<net ") != std::string::npos;
}

// Common parsing utilities
std::string TrimWhitespace(const std::string& str);
std::string TrimQuotes(const std::string& str);
std::pair<int64_t, int64_t> ParseDimensionRange(const std::string& range_str, const std::string& tensor_name);

}  // namespace utils
}  // namespace openvino_ep
}  // namespace onnxruntime
