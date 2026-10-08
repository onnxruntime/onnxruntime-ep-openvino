// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <set>
#include <regex>
#include <algorithm>
#include <format>
#include <fstream>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "ov_plugin_utils.h"
#include "ov_common_utils.h"
#include "ov_constants.h"
#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

static void ParseInnerMap(const Ort::Logger& logger, const nlohmann::json& json_map, ov::AnyMap& inner_map, size_t level = 0) {
  const size_t max_levels = 8;
  if (level >= max_levels) {
    ORT_CXX_API_THROW(std::format("load_config can have only up to {} levels of nested maps. Current level = {}", max_levels, level), ORT_INVALID_ARGUMENT);
  }

  if (!json_map.is_object()) {
    ORT_CXX_API_THROW("ParseInnerMap: Expected an object as input", ORT_INVALID_ARGUMENT);
  }

  for (auto& [inner_key, inner_value] : json_map.items()) {
    if (inner_value.is_string()) {
      inner_map[inner_key] = ov::Any(inner_value.get<std::string>());
    } else if (inner_value.is_number_integer()) {
      inner_map[inner_key] = ov::Any(inner_value.get<int64_t>());
    } else if (inner_value.is_number_float()) {
      inner_map[inner_key] = ov::Any(inner_value.get<double>());
    } else if (inner_value.is_boolean()) {
      inner_map[inner_key] = ov::Any(inner_value.get<bool>());
    } else if (inner_value.is_object()) {
      auto inner_inner_map = ov::AnyMap();
      ParseInnerMap(logger, inner_value, inner_inner_map, level + 1);
      inner_map[inner_key] = std::move(inner_inner_map);
    } else {
      ORT_CXX_API_THROW(std::format("load_config: unsupported JSON value type={}, for key={}", std::string(inner_value.type_name()), inner_key), ORT_INVALID_ARGUMENT);
    }
  }
}

void ParsePluginLoadConfigOption(const Ort::Logger& logger, const std::string& config_str, ConfigMap& target_map) {
  if (config_str.empty()) {
    ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING, "Empty OV Config Map passed. Skipping load_config option parsing.");
    target_map = {};
    return;
  }

  std::string json_content = config_str;

  // Check if config_str is a file path by looking for file extension
  if (config_str.ends_with(".json")) {
    try {
      std::ifstream file(config_str);
      if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        json_content = buffer.str();
        file.close();

        if (json_content.empty()) {
          ORT_CXX_API_THROW(std::format("Config file is empty: {}", config_str), ORT_INVALID_ARGUMENT);
        }
      }
    } catch (const Ort::Exception&) {
      throw;
    } catch (const std::exception& e) {
      ORT_CXX_API_THROW(std::format("Error reading config file {}: {}", config_str, e.what()), ORT_INVALID_ARGUMENT);
    }
  }

  std::stringstream input_str_stream(json_content);
  try {
    nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

    if (!json_config.is_object()) {
      ORT_CXX_API_THROW("Invalid JSON structure: Expected an object at the root.", ORT_INVALID_ARGUMENT);
    }

    for (const auto& [key, value] : json_config.items()) {
      ov::AnyMap inner_map;

      // Ensure that the value for each device is an object (PROPERTY -> VALUE)
      if (!value.is_object()) {
        ORT_CXX_API_THROW("Invalid JSON structure: Expected an object for device properties.", ORT_INVALID_ARGUMENT);
      }

      ParseInnerMap(logger, value, inner_map);
      target_map[key] = std::move(inner_map);
    }
  } catch (const nlohmann::json::parse_error& e) {
    // Handle syntax errors in JSON
    ORT_CXX_API_THROW(std::format("JSON parsing error: {}", e.what()), ORT_INVALID_ARGUMENT);
  } catch (const nlohmann::json::type_error& e) {
    // Handle invalid type accesses
    ORT_CXX_API_THROW(std::format("JSON type error: {}", e.what()), ORT_INVALID_ARGUMENT);
  } catch (const Ort::Exception&) {
    throw;
  } catch (const std::exception& e) {
    ORT_CXX_API_THROW(std::format("Error parsing load_config Map: {}", e.what()), ORT_INVALID_ARGUMENT);
  }
}

void ParsePluginReshapeInputOption(const std::string& config_str, ReshapeMap& reshape_map) {
  // Return empty map for empty input
  if (config_str.empty()) {
    ORT_CXX_API_THROW(std::format("reshape_input parameter: Invalid input shape definition \"{}\"", config_str), ORT_INVALID_ARGUMENT);
  }

  try {
    // Regular expressions for parsing
    const std::regex tensor_pattern(R"(([^\[\],]+)\s*\[(.*?)\])");  // e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
    const std::regex dimension_pattern(R"(\s*([^,\s]+)\s*)");

    // Find all tensor shape definitions using regex
    auto tensor_begin = std::sregex_iterator(
        config_str.begin(),
        config_str.end(),
        tensor_pattern);
    auto tensor_end = std::sregex_iterator();

    // If no matches found, throw error
    if (tensor_begin == tensor_end) {
      ORT_CXX_API_THROW(std::format("reshape_input parameter: Invalid input shape definition format: {}", config_str), ORT_INVALID_ARGUMENT);
    }

    // Process each tensor definition e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
    for (std::sregex_iterator i = std::move(tensor_begin); i != tensor_end; ++i) {
      std::smatch tensor_match = *i;

      // Extract tensor name and trim whitespace
      std::string tensor_name = tensor_match[1].str();  // Group 1: tensor name e.g. "input_1"
      tensor_name = onnxruntime::openvino_ep::utils::TrimWhitespace(tensor_name);

      if (tensor_name.empty()) {
        ORT_CXX_API_THROW("reshape_input parameter: Empty tensor name provided in reshape_input parameter", ORT_INVALID_ARGUMENT);
      }

      // Extract dimensions string
      std::string dimensions_str = tensor_match[2].str();  // Group 2: dimensions string [e.g. "1..5, 2, 3..4"]
      std::vector<ov::Dimension> dimensions;

      // Find all dimension e.g. "1..5", "2", "3..4" using regex
      auto dim_begin = std::sregex_iterator(
          dimensions_str.begin(),
          dimensions_str.end(),
          dimension_pattern);
      auto dim_end = std::sregex_iterator();

      if (dim_begin == dim_end) {
        ORT_CXX_API_THROW(std::format("reshape_input parameter: Empty tensor dimensions for tensor: {}", tensor_name), ORT_INVALID_ARGUMENT);
      }

      // Process each dimension
      for (std::sregex_iterator j = std::move(dim_begin); j != dim_end; ++j) {
        std::smatch dim_match = *j;
        std::string dim_value = dim_match[1].str();

        // Check if dimension is a range
        size_t range_separator_pos = dim_value.find("..");
        if (range_separator_pos != std::string::npos) {
          // Parse range
          auto range_pair = onnxruntime::openvino_ep::utils::ParseDimensionRange(dim_value, tensor_name);
          dimensions.push_back(ov::Dimension(range_pair.first, range_pair.second));
        } else {
          // Parse single value
          bool is_valid_integer = !dim_value.empty() &&
                                  std::all_of(dim_value.begin(), dim_value.end(), [](char c) {
                                    return std::isdigit(static_cast<unsigned char>(c));
                                  });

          if (!is_valid_integer) {
            ORT_CXX_API_THROW(std::format("reshape_input parameter: Invalid dimension value: '{}' for tensor: {}", dim_value, tensor_name), ORT_INVALID_ARGUMENT);
          }

          dimensions.push_back(std::stoi(dim_value));
        }
      }

      // Store parsed shape in result map
      reshape_map[tensor_name] = ov::PartialShape(std::move(dimensions));
    }
  } catch (const Ort::Exception&) {
    throw;
  } catch (const std::exception& e) {
    ORT_CXX_API_THROW(std::format("reshape_input parameter: Error parsing reshape_input: {}", e.what()), ORT_INVALID_ARGUMENT);
  }
}

static OrtStatus* ParseInt32CommaList(const OrtApi& ort_api, std::string_view input, const std::string& index_type, std::vector<int32_t>& int_list) {
  auto remaining = input;
  while (!remaining.empty()) {
    const auto comma_pos = remaining.find(',');
    const auto token = remaining.substr(0, comma_pos);
    const auto first_digit_pos = token.find_first_not_of(" \t");  // skip leading spaces or tabs
    const auto last_digit_pos = token.find_last_not_of(" \t");

    // Check if token is empty or only whitespace
    if (first_digit_pos == std::string_view::npos) {
      // Skip empty tokens (handles trailing commas like "1,2,")
      if (comma_pos == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(comma_pos + 1);
      continue;
    }

    const auto trimmed_token = token.substr(first_digit_pos, last_digit_pos - first_digit_pos + 1);

    int32_t parsed_value = 0;
    const auto ret = std::from_chars(trimmed_token.data(), trimmed_token.data() + trimmed_token.size(), parsed_value);

    if (ret.ec != std::errc()) {
      if (ret.ec == std::errc::result_out_of_range)
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    std::format("kvcache_reorder {} exceeds INT32_MAX: {}", index_type, std::string(token)).c_str());

      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  std::format("Failed to parse kvcache_reorder {}: {}", index_type, std::string(token)).c_str());
    }
    if (parsed_value < 0) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  std::format("kvcache_reorder {} cannot be negative: {}", index_type, std::string(token)).c_str());
    }
    int_list.push_back(parsed_value);
    if (comma_pos == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(comma_pos + 1);
  }

  return nullptr;
}

OrtStatus* ParseSrcAndDstIntCommaLists(const OrtApi& ort_api, const std::string& input, std::vector<int32_t>& src_list, std::vector<int32_t>& dst_list) {
  // Convert kvcache_reorder value format "1,2,3;4,5,6" into two vectors
  // src_indices = [1,2,3], dst_indices = [4,5,6]
  // Special case: ";" or "" means empty lists (no-op/clear operation)

  if (input.empty() || input == ";") {
    return nullptr;
  }

  const std::string trimmed_input = openvino_ep::utils::TrimWhitespace(input);

  if (trimmed_input.empty() || trimmed_input == ";") {
    return nullptr;
  }

  const auto delimiter_pos = trimmed_input.find(';');
  if (delimiter_pos == std::string::npos) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "kvcache_reorder value format is incorrect, expected 'x1,x2;x3,x4'");
  }

  // Check for multiple semicolons
  if (trimmed_input.find(';', delimiter_pos + 1) != std::string::npos) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "kvcache_reorder value format is incorrect: only one semicolon delimiter is allowed (expected 'x1,x2;x3,x4')");
  }

  std::string_view src_string = std::string_view(trimmed_input).substr(0, delimiter_pos);
  std::string_view dst_string = std::string_view(trimmed_input).substr(delimiter_pos + 1);

  if (src_string.empty() && dst_string.empty()) {
    return nullptr;
  }

  if (src_string.empty() != dst_string.empty()) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "kvcache_reorder source and destination lists must both be empty or both be non-empty");
  }

  OVEP_RETURN_IF_ERROR(ParseInt32CommaList(ort_api, src_string, "src_index", src_list));
  OVEP_RETURN_IF_ERROR(ParseInt32CommaList(ort_api, dst_string, "dst_index", dst_list));

  return nullptr;
}

std::optional<StringPairVector> ParseStringPairVector(const std::string& raw_value) {
  StringPairVector parsed_ops;
  auto without_quotes = openvino_ep::utils::TrimQuotes(raw_value);

  if (without_quotes.empty())
    return parsed_ops;

  for (auto&& pair : std::ranges::split_view(without_quotes, ',')) {
    auto pair_str = openvino_ep::utils::TrimWhitespace(std::string(pair.begin(), pair.end()));

    if (pair_str.empty())
      continue;  // Skip empty segments (e.g., "a:b,,c:d,    ,abc")

    const size_t colon_pos = pair_str.find(':');

    const std::string domain = colon_pos == std::string::npos ? "" : openvino_ep::utils::TrimWhitespace(pair_str.substr(0, colon_pos));
    const std::string op_name = openvino_ep::utils::TrimWhitespace(colon_pos == std::string::npos ? pair_str : pair_str.substr(colon_pos + 1));

    if (op_name.empty())
      return std::nullopt;

    parsed_ops.emplace_back(op_name, domain);
  }

  return parsed_ops;
}

bool ShouldRecompile(const char* ov_exception) noexcept {
  try {
    std::string ov_exception_string(ov_exception);
    std::smatch matches;
    std::regex error_message_pattern(R"(\bZE_\w*\b)");
    if (!std::regex_search(ov_exception_string, matches, error_message_pattern)) {
      return false;
    }
    std::regex error_code_pattern("code 0x([0-9a-fA-F]+)");
    uint32_t error_code{0};
    if (std::regex_search(ov_exception_string, matches, error_code_pattern)) {
      std::from_chars(&(*matches[1].first), &(*matches[1].second), error_code, 16);
      return error_code == 0x7800000f /* ZE_RESULT_ERROR_INVALID_NATIVE_BINARY */;
    }
  } catch (...) {
    // don't want to throw since we intend to call from an exception handler already.
  }
  return false;
}

namespace features {
feature<NotProduction> debug(env_string_debug, false);
feature<NotProduction> enable_plugin_unloading(env_string_enable_plugin_unloading, false);
feature<NotProduction> disable_arena(env_string_disable_arena, false);
feature<NotProduction> cpu_va_for_igpu(env_string_cpu_va_for_igpu, true);
feature<NotProduction> disable_device_specific_allocators(env_string_device_specific_allocators, false);
ops_option<NotProduction> accept_ops(env_string_accept_ops);
ops_option<NotProduction> reject_ops(env_string_reject_ops);
}  // namespace features

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
