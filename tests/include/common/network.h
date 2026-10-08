// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ort_helper.h"
#include "path_utils.h"
#include <onnxruntime/onnxruntime_session_options_config_keys.h>
#include <cstdint>
#include <span>

namespace fs = std::filesystem;

namespace ovep_tests {

///////////////////////////////////////////////////////////////////////
struct network_t {
  std::string name_{};
  std::vector<uint8_t> buffer_;

  std::map<std::string, std::string> ort_provider_options_{};
  std::map<std::string, std::string> ort_provider_options_load_config_{};
  std::map<std::string, int64_t> ort_free_dim_override_;
  std::map<std::string, std::string> ort_session_config_;
  int32_t ort_graph_optimization_level_ = -1;
  std::map<std::string, std::string> ort_ep_dynamic_options_;
  bool ort_enable_io_binding_ = false;

  // Weightless source model bytes. The caller keeps them alive for the session, matching
  // OrtApi::SessionOptionsSetWeightlessSourceModelBuffer.
  std::span<const uint8_t> weightless_source_model_buffer_{};

  ///////////////////////////////////////////////////////////////////////
  void set_cpu_fallback(bool enable) {
    ort_session_config_[kOrtSessionOptionsDisableCPUEPFallback] = enable ? "0" : "1";
  }

  ///////////////////////////////////////////////////////////////////////
  fs::path model_path(void) const {
    fs::path model_name = path_from_utf8(name_);

    if (!model_name.is_relative()) {
      return model_name;
    }

    const auto model_dir_env = get_environment_variable("OVEP_TESTS_MODEL_DIR");
    if (model_dir_env.has_value()) {
      model_name = fs::path(*model_dir_env) / model_name;
    }

    fs::path model_dir = model_name.parent_path();
    if (model_dir.empty()) {
      model_dir = fs::current_path();
    }

    auto model_name_prefix = path_to_utf8(model_name.filename());

    std::vector<fs::path> matching_files;
    for (const auto& itr : fs::directory_iterator(model_dir)) {
      if (fs::is_regular_file(itr.path())) {
        const std::string filename = path_to_utf8(itr.path().filename());
        if (filename.compare(0, model_name_prefix.length(), model_name_prefix) == 0) {
          matching_files.push_back(itr.path());
        }
      }
    }

    if (matching_files.empty()) {
      OVEP_TESTS_THROW(
          "No files found matching the network name %s in model dir %s",
          model_name_prefix.c_str(),
          path_to_utf8(model_dir).c_str());
    } else if (matching_files.size() > 1) {
      OVEP_TESTS_THROW(
          "Matched more than a single model to network name %s in model dir: %s",
          model_name_prefix.c_str(),
          path_to_utf8(model_dir).c_str());
    }

    return matching_files[0];
  }

  ///////////////////////////////////////////////////////////////////////
  static network_t create(
      fs::path model_path,
      session_model_load_mode_t load_mode = session_model_load_mode_t::file) {
    network_t network{};
    switch (load_mode) {
      case session_model_load_mode_t::file: {
        network.name_ = path_to_utf8(fs::absolute(model_path));
        return network;
      }
      case session_model_load_mode_t::memory: {
        data_utils::read_binary_file(model_path, network.buffer_);
        return network;
      }
      default:
        throw std::invalid_argument("Unknown load mode");
    }
  };
};

}  // namespace ovep_tests