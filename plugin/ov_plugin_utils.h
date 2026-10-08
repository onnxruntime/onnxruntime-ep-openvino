// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <map>
#include <optional>
#include <regex>

#include "openvino/core/any.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_utils.h"
#include "ov_version.h"
#include "ov_plugin_ver.h"
#include "ov_common_utils.h"
#include "ov_constants.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace onnxruntime {
namespace openvino_ep_plugin {

using ConfigMap = std::map<std::string, ov::AnyMap>;
using ReshapeMap = std::map<std::string, ov::PartialShape>;

void ParsePluginLoadConfigOption(const Ort::Logger& logger, const std::string& config_str, ConfigMap& target_map);
void ParsePluginReshapeInputOption(const std::string& config_str, ReshapeMap& reshape_map);
OrtStatus* ParseSrcAndDstIntCommaLists(const OrtApi& ort_api, const std::string& input, std::vector<int32_t>& src_list, std::vector<int32_t>& dst_list);

// example: $env:ORT_OPENVINO_ACCEPT_OPS="com.microsoft:GQA,  com.default:VNorm, HNorm, "
using StringPairVector = std::vector<std::pair<std::string, std::string>>;
std::optional<StringPairVector> ParseStringPairVector(const std::string& raw_value);

#ifdef PRODUCTION
static constexpr std::bool_constant<true> Production;
#else
static constexpr std::bool_constant<false> Production;
#endif

static constexpr std::bool_constant<!Production> NotProduction;

#ifdef RELEASE
static constexpr std::bool_constant<true> Release;
#else
static constexpr std::bool_constant<false> Release;
#endif

static constexpr std::bool_constant<!Release> NotRelease;

template <typename T, bool... Conditions>
struct option {
  using Type = T;
  using ConvertFcn = std::optional<Type> (*)(const std::string&);

  static std::optional<Type> convert_string_to_type(const std::string& val) {
    if constexpr (std::is_convertible_v<Type, int>) {
      return std::stoi(val);
    } else if constexpr (std::is_convertible_v<Type, long>) {
      return std::stol(val);
    } else if constexpr (std::is_convertible_v<Type, long long>) {
      return std::stoll(val);
    } else if constexpr (std::is_convertible_v<Type, unsigned long>) {
      return std::stoul(val);
    } else if constexpr (std::is_convertible_v<Type, unsigned long long>) {
      return std::stoull(val);
    } else if constexpr (std::is_convertible_v<Type, float>) {
      return std::stof(val);
    } else if constexpr (std::is_convertible_v<Type, double>) {
      return std::stod(val);
    } else if constexpr (std::is_convertible_v<Type, long double>) {
      return std::stold(val);
    } else {
      static_assert(false, "Unsupported conversion");
    }

    return std::nullopt;
  };

  option() = delete;

  option(const std::string_view name,
         Type&& init,
         const ConvertFcn convert_fcn = convert_string_to_type) noexcept : convert_fcn_(convert_fcn),
                                                                           name_(name),
                                                                           value_(std::move(init)) {
    get_latest_value();
  }

  const Type& get_latest_value() noexcept {
    if constexpr (supported_) {
      try {
        auto env_var_value = GetEnvironmentVar(name_);
        if (auto converted_value = convert_fcn_(env_var_value)) {
          value_ = converted_value.value();
        }
      } catch (const std::exception&) {
        // Ignore conversion errors and use default value
      }
    }
    return value_;
  }

  constexpr bool is_supported() {
    return supported_;
  }

  const ConvertFcn convert_fcn_;
  const std::string_view name_;
  Type value_;
  static constexpr bool supported_ = std::conjunction_v<std::bool_constant<Conditions>...>;
};

namespace features {

template <bool... Conditions>
struct feature : public option<bool, Conditions...> {
  using Parent = option<bool, Conditions...>;
  using Type = Parent::Type;
  static std::optional<Type> convert_string_to_bool(const std::string& val) {
    if (val == "1") {
      return true;
    } else if (val == "0") {
      return false;
    } else {
      return std::nullopt;
    }
  };

  feature() = delete;

  feature(const std::string_view name, bool init) noexcept : Parent(name, std::move(init), convert_string_to_bool) {}

  Type is_enabled() noexcept { return Parent::value_; }
};

template <bool... Conditions>
struct ops_option : public option<StringPairVector, Conditions...> {
  using Parent = option<StringPairVector, Conditions...>;
  using Type = typename Parent::Type;

  explicit ops_option(std::string_view name) noexcept : Parent(name, Type{}, &ops_option::Convert) {}

 private:
  // Parses comma-separated domain:op_name pairs; the domain can be empty.
  static std::optional<StringPairVector> Convert(const std::string& raw_value) {
    return ParseStringPairVector(raw_value);
  }
};

extern feature<NotProduction> debug;
extern feature<NotProduction> enable_plugin_unloading;
extern feature<NotProduction> disable_arena;
extern feature<NotProduction> cpu_va_for_igpu;
extern feature<NotProduction> disable_device_specific_allocators;
extern ops_option<NotProduction> accept_ops;
extern ops_option<NotProduction> reject_ops;

}  // namespace features

bool ShouldRecompile(const char* ov_exception) noexcept;

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
