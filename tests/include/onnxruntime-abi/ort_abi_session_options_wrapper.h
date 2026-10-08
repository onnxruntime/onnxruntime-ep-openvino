// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ovep_tests {
namespace onnxruntime_abi {
///////////////////////////////////////////////////////////////////////
template <typename fixture_t>
struct session_options_wrapper : Ort::SessionOptions {
  using device_t = typename fixture_t::device_wrapper_t;
  using custom_ops_t = typename fixture_t::custom_ops_wrapper_t;

  device_t device_;
  std::shared_ptr<custom_ops_t> custom_ops_;

  OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(session_options_wrapper);
  OVEP_TESTS_DEFAULT_MOVE(session_options_wrapper);

  ///////////////////////////////////////////////////////////////////////////
  session_options_wrapper(
      device_t&& device,
      std::map<std::string, std::string> session_config = {},
      std::map<std::string, std::string> provider_options = {},
      std::map<std::string, std::string> provider_options_load_config = {})
      : Ort::SessionOptions{}, device_(std::move(device)) {
    add_session_config_entry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
    add_session_config_entry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");

    if (device_.ort_execution_provider_ != ort_execution_provider_t::cpu) {
      AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }

    for (const auto& [key, value] : session_config) {
      add_session_config_entry(key.c_str(), value.c_str());
    }

    set_graph_optimization_level(
        device_.ort_execution_provider_ == ort_execution_provider_t::ov ? ORT_DISABLE_ALL : ORT_ENABLE_ALL);

    if (device_.ort_execution_provider_ != ort_execution_provider_t::cpu) {
      set_execution_mode(ORT_SEQUENTIAL);
      disable_mem_pattern();

      // If the device contains a policy string, use it (policy-only device).
      if (device_.ort_selection_policy_.has_value()) {
        if (!provider_options.empty() || !provider_options_load_config.empty()) {
          OVEP_TESTS_THROW(
              "Cannot specify both EpDeviceSelectionPolicy and provider options - provider options cannot be "
              "expressed in ORT api");
        }
        set_execution_provider_selection_policy(device_.ort_selection_policy_.value());
      } else {
        // No policy on device => expect an enumerated ep_device_
        if (device_.ep_device_.empty()) {
          OVEP_TESTS_THROW("Either EpDeviceSelectionPolicy or ep device must be specified");
        }

        // Append provider when ep_device_ is present
        append_execution_provider(device_, provider_options, provider_options_load_config);
      }
    }

    custom_ops_ = std::make_shared<custom_ops_t>(*this);
  }

  ///////////////////////////////////////////////////////////////////////
  session_options_wrapper(const device_t& device, const network_t& network)
      : session_options_wrapper(
            device.name_,
            network.ort_session_config_,
            network.ort_provider_options_,
            network.ort_provider_options_load_config_) {
    for (const auto& [name, value] : network.ort_free_dim_override_) {
      add_free_dim_override(name.c_str(), value);
    }

    if (!network.weightless_source_model_buffer_.empty()) {
      set_weightless_source_model_buffer(network.weightless_source_model_buffer_);
    }

    set_graph_optimization_level(
        network.ort_graph_optimization_level_ >= 0
            ? static_cast<GraphOptimizationLevel>(network.ort_graph_optimization_level_)
            : (device_.ort_execution_provider_ == ort_execution_provider_t::ov ? ORT_DISABLE_ALL
                                                                               : ORT_ENABLE_ALL));
  }

  ///////////////////////////////////////////////////////////////////////
  virtual ~session_options_wrapper(void) {}

 private:
  ///////////////////////////////////////////////////////////////////////
  inline void set_execution_mode(ExecutionMode mode) {
    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetExecutionMode( ort_session_options: " << this
                         << ", mode: " << to_string(mode) << " ): {";

    SetExecutionMode(mode);

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetExecutionMode: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void disable_mem_pattern(void) {
    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::DisableMemPattern( ort_session_options: " << this << " ): {";

    DisableMemPattern();

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::DisableMemPattern: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline std::string serialize_config(
      const std::map<std::string, std::string>& config,
      const std::string& device_name) {
    nlohmann::json json;

    for (auto& [key, val] : config) {
      json[device_name][key] = val;
    }

    return json.dump();
  }

  ///////////////////////////////////////////////////////////////////////
  inline void append_execution_provider(
      const device_t& device,
      const std::map<std::string, std::string>& config,
      const std::map<std::string, std::string>& load_config) {
    std::unordered_map<std::string, std::string> options;

    for (const auto& [key, val] : config) {
      options.emplace(key, val);
    }

    if (!load_config.empty()) {
      options["load_config"] = serialize_config(load_config, device.name_);
    }

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::AppendExecutionProvider_V2: {";

    AppendExecutionProvider_V2(env_, {device.ep_device_.front()}, options);

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::AppendExecutionProvider_V2: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void add_free_dim_override(const char* name, int64_t value) {
    OVEP_TESTS_LOG_DEBUG << "OrtApi::AddFreeDimensionOverrideByName( ort_session_options: " << this
                         << ", name: " << name << ", value: " << value << " ): {";

    Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(GetUnowned(), name, value));

    OVEP_TESTS_LOG_DEBUG << "OrtApi::AddFreeDimensionOverrideByName: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void set_weightless_source_model_buffer(std::span<const uint8_t> buffer) {
    OVEP_TESTS_LOG_DEBUG << "OrtApi::SessionOptionsSetWeightlessSourceModelBuffer( ort_session_options: " << this
                         << ", bytes: " << buffer.size() << " ): {";

    Ort::ThrowOnError(Ort::GetApi().SessionOptionsSetWeightlessSourceModelBuffer(
        GetUnowned(), buffer.data(), buffer.size()));

    OVEP_TESTS_LOG_DEBUG << "OrtApi::SessionOptionsSetWeightlessSourceModelBuffer: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void add_session_config_entry(const char* key, const char* value) {
    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::AddConfigEntry( ort_session_options: " << this << ", key: " << key
                         << ", value: " << value << " ): {";

    AddConfigEntry(key, value);

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::AddConfigEntry: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void set_graph_optimization_level(GraphOptimizationLevel level) {
    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetGraphOptimizationLevel( ort_session_options: " << this
                         << ", mode: " << to_string(level) << " ): {";

    SetGraphOptimizationLevel(level);

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetGraphOptimizationLevel: ...OK }";
  }

  ///////////////////////////////////////////////////////////////////////
  inline void set_execution_provider_selection_policy(OrtExecutionProviderDevicePolicy policy) {
    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetEpSelectionPolicy( ort_session_options: " << this
                         << ", policy: " << to_string(policy) << " ): {";

    SetEpSelectionPolicy(policy);

    OVEP_TESTS_LOG_DEBUG << "OrtSessionOptions::SetEpSelectionPolicy: ...OK }";
  }
};
}  // namespace onnxruntime_abi
}  // namespace ovep_tests