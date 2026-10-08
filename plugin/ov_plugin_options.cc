// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ort_api_version.h"
#include "ov_plugin_options.h"
#include "ov_plugin_utils.h"
#include "ov_utils.h"
#include "onnxruntime/onnxruntime_session_options_config_keys.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <format>

namespace onnxruntime {
namespace openvino_ep_plugin {

template <typename T>
static T GetSessionConfigEntryOrDefault(Ort::ConstSessionOptions session_options,
                                        const char* config_key, T default_val) {
  std::string string_val = session_options.GetConfigEntryOrDefault(config_key, std::string{});

  if (string_val.empty()) {
    return default_val;
  }

  if constexpr (std::is_same_v<T, std::string>) {
    return string_val;
  } else if constexpr (std::is_same_v<T, int>) {
    try {
      return std::stoi(string_val);
    } catch (const std::exception& ex) {
      ORT_CXX_API_THROW(std::format("Invalid integer value '{}' for config key '{}': {}", string_val, config_key, ex.what()),
                        ORT_INVALID_ARGUMENT);
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    return string_val == "1";
  } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
    return PathFromUtf8(string_val);
  } else {
    static_assert(false, "unsupported type");
  }
}

// Lets an app hand over a source model it holds in memory - one that is encrypted on disk and
// decrypted by the app.
static std::span<const uint8_t> GetWeightlessSourceModelBuffer(Ort::ConstSessionOptions session_options) {
  std::span<const uint8_t> buffer{};
  OrtVersionGate<29>([&] {
    const void* data = nullptr;
    size_t length = 0;
    Ort::ThrowOnError(
        Ort::GetEpApi().SessionOptionsGetWeightlessSourceModelBuffer(session_options, &data, &length));
    if (data != nullptr && length > 0) {
      buffer = {static_cast<const uint8_t*>(data), length};
    }
  });
  return buffer;
}

EpContextSessionConfig OpenVINOEpPluginOptions::ParseEpContextOptions(Ort::ConstSessionOptions session_options) {
  EpContextSessionConfig ctx{
      .enable_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionEpContextEnable, false),
      .embed_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionEpContextEmbedMode, false),
      .share_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionShareEpContexts, false),
      .stop_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionStopShareEpContexts, false),
      .path_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionEpContextFilePath, std::filesystem::path{}),
      .weightless_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionEpEnableWeightless, false),
      .source_model_path_ = GetSessionConfigEntryOrDefault(session_options, kOrtSessionOptionEpContextSourceModelPath, std::filesystem::path{}),
      .source_model_buffer_ = GetWeightlessSourceModelBuffer(session_options),
  };
  // Disallow embed mode if sharing is enabled.
  if (ctx.enable_) {
    OVEP_ENFORCE(!ctx.embed_ || !ctx.share_,
                 "Invalid EP context configuration: ", kOrtSessionOptionEpContextEmbedMode,
                 " must be 0 if, ", kOrtSessionOptionShareEpContexts, " is 1.");
  }

  if (!ctx.source_model_path_.empty()) {
    std::string extension = ToUtf8String(ctx.source_model_path_.extension());
    std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return std::tolower(c); });
    OVEP_ENFORCE(extension == ".onnx",
                 "Invalid ", kOrtSessionOptionEpContextSourceModelPath,
                 ": expected a .onnx model, got ", ToUtf8String(ctx.source_model_path_));
  }
  return ctx;
}

OpenVINOEpPluginOptions::OpenVINOEpPluginOptions(std::string ep_name, const Ort::Logger& logger,
                                                 Ort::ConstSessionOptions session_options)
    : ep_ctx_(ParseEpContextOptions(session_options)),
      ep_name_(std::move(ep_name)) {
  debug_enabled_ = features::debug.get_latest_value();

  ParseProviderOptions(logger, session_options);

  if (ep_ctx_.enable_) {
    provider_options_.load_config_["NPU"].try_emplace("NPU_BYPASS_UMD_CACHING", "YES");
  }
}

void OpenVINOEpPluginOptions::ParseProviderOptions(const Ort::Logger& logger, Ort::ConstSessionOptions session_options) {
  std::string lower_ep_name = ep_name_;
  std::ranges::transform(lower_ep_name, lower_ep_name.begin(), [](unsigned char c) { return std::tolower(c); });
  const std::string provider_prefix = "ep." + lower_ep_name + ".";

  // Define valid keys for the OpenVINO EP provider options
  const auto& valid_keys = OpenVINOEpProviderOptions::GetValidProviderKeys();

  // Helper to get option value from session config
  auto get_option_value = [&](std::string_view option_name) -> std::optional<std::string> {
    auto value = GetSessionConfigEntryOrDefault(session_options, (provider_prefix + std::string(option_name)).c_str(), std::string{});
    return value.empty() ? std::nullopt : std::optional{std::move(value)};
  };

  // Helper to parse boolean values with case-insensitive comparison
  auto parse_bool = [](std::string_view value) -> std::optional<bool> {
    std::string lower(value);
    std::ranges::transform(lower, lower.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lower == "true" || lower == "1") return true;
    if (lower == "false" || lower == "0") return false;
    return std::nullopt;
  };

  // Process each valid option if provided
  for (const auto& key : valid_keys) {
    auto value_opt = get_option_value(key);
    if (!value_opt) continue;  // Option not provided

    const std::string& value = *value_opt;

    if (key == "enable_causallm") {
      if (auto bool_val = parse_bool(value)) {
        provider_options_.enable_causallm_ = *bool_val;
      } else {
        const std::string error_msg = "Invalid boolean value '" + value + "' for option '" + key +
                                      "'. Valid values are: 'true', 'false', '1', '0' (case insensitive)";
        ORT_CXX_API_THROW(error_msg, ORT_INVALID_ARGUMENT);
      }
    } else if (key == "hint_enable_gqa") {
      if (auto bool_val = parse_bool(value)) {
        provider_options_.hint_enable_gqa_ = *bool_val;
      } else {
        const std::string error_msg = "Invalid boolean value '" + value + "' for option '" + key +
                                      "'. Valid values are: 'true', 'false', '1', '0'";
        ORT_CXX_API_THROW(error_msg, ORT_INVALID_ARGUMENT);
      }
    } else if (key == "dump_subgraphs") {
      // dump_subgraphs expects a dump output path.
      provider_options_.dump_subgraphs_path_ = PathFromUtf8(value);
    } else if (key == "load_config") {
      // Parse load_config using the plugin-specific utility function
      ParsePluginLoadConfigOption(logger, value, provider_options_.load_config_);
    } else if (key == "reshape_input") {
      // Parse reshape_input option using the plugin-specific utility function
      ParsePluginReshapeInputOption(value, provider_options_.reshape_input_);
    }
  }

  static const std::vector<std::string> advise_on_device_selection_keys = {"device_type", "device_id", "device_luid"};
  for (const auto& key : advise_on_device_selection_keys) {
    auto value_opt = get_option_value(key);
    if (!value_opt) continue;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, std::format("Provider option {} is ignored. Please pass one of the enumerated EP device to AppendExecutionProvider_V2 for device selection.", key).c_str());
  }

  static const std::unordered_map<std::string, std::string> advise_on_load_config_entries = {
      {"cache_dir", "\"CACHE_DIR\": \"<filesystem_path>\""},
      {"precision", "\"INFERENCE_PRECISION_HINT\": \"F32\""},
      {"num_of_threads", "\"INFERENCE_NUM_THREADS\": \"1\""},
      {"num_streams", "\"NUM_STREAMS\": \"1\""},
      {"model_priority", "\"MODEL_PRIORITY\": \"LOW\""},
      {"enable_opencl_throttling", "\"GPU\": {\"PLUGIN_THROTTLE\": \"1\"}"},
      {"enable_qdq_optimizer", "\"NPU\": {\"NPU_QDQ_OPTIMIZATION\": \"YES\"}"}};

  for (const auto& [key, advise_entry] : advise_on_load_config_entries) {
    auto value_opt = get_option_value(key);
    if (!value_opt) continue;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, std::format("Provider option {} is ignored. Please set it via the load_config as: {}", key, advise_entry).c_str());
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
