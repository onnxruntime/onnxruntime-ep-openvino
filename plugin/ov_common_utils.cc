// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <istream>
#include <string>
#include <algorithm>

#include "ov_common_utils.h"

namespace onnxruntime {
namespace openvino_ep {
namespace utils {

bool IsModelStreamXML(std::istream& model_stream) {
  std::streampos originalPos = model_stream.tellg();

  // first, get the total size of model_stream in bytes
  model_stream.seekg(0, std::ios::end);
  auto end_pos = model_stream.tellg();
  //  Restore the stream position
  model_stream.seekg(originalPos);
  auto total_size = end_pos - originalPos;

  // Choose 32 bytes to hold content of:
  // '<?xml version-"1.0"?> <net '
  const std::streamsize header_check_len = 32;
  if (total_size < header_check_len) {
    return false;
  }

  // read 32 bytes into header
  std::string header(header_check_len, '\0');
  model_stream.read(&header[0], header_check_len);
  // Clear any read errors
  model_stream.clear();
  // Restore the stream position
  model_stream.seekg(originalPos);

  return IsXmlHeader(header);
}

// Helper function to trim whitespace from a string
std::string TrimWhitespace(const std::string& str) {
  const std::string whitespace = " \t\n\r\f\v";
  size_t start = str.find_first_not_of(whitespace);

  if (start == std::string::npos) {
    return "";
  }

  size_t end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}

// optionally strip quotes from the first and the last character
std::string TrimQuotes(const std::string& str) {
  if (str.size() > 2 && str.front() == '"' && str.back() == '"') {
    return str.substr(1, str.size() - 2);
  }

  return str;
}

// Helper function to parse dimension range (e.g. "1..5")
std::pair<int64_t, int64_t> ParseDimensionRange(const std::string& range_str, const std::string& tensor_name) {
  size_t range_separator_pos = range_str.find("..");
  if (range_separator_pos == std::string::npos) {
    throw std::invalid_argument("Invalid dimension range format: " + range_str);
  }

  std::string range_start_str = TrimWhitespace(range_str.substr(0, range_separator_pos));
  std::string range_end_str = TrimWhitespace(range_str.substr(range_separator_pos + 2));

  // Validate range values
  if (range_start_str.empty() || range_end_str.empty() ||
      !std::all_of(range_start_str.begin(), range_start_str.end(), ::isdigit) ||
      !std::all_of(range_end_str.begin(), range_end_str.end(), ::isdigit)) {
    throw std::invalid_argument("Invalid dimension range format: '" + range_str + "' for tensor: " + tensor_name);
  }

  int64_t range_start = std::stoll(range_start_str);
  int64_t range_end = std::stoll(range_end_str);

  if (range_start > range_end) {
    throw std::invalid_argument("Invalid dimension range (start > end): " + range_str + " for tensor: " + tensor_name);
  }

  return std::make_pair(range_start, range_end);
}

}  // namespace utils
}  // namespace openvino_ep
}  // namespace onnxruntime
