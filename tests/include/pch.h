// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <array>
#include <vector>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <thread>
#include <random>
#include <climits>
#include <functional>
#include <filesystem>
#include <format>
#include <condition_variable>
#include <atomic>
#include <regex>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <queue>
#include <unordered_set>
#include <optional>
#include <span>
#include <ranges>

#ifdef _WIN32
#include <Windows.h>
#include <timeapi.h>
#include <psapi.h>
#include <Pdh.h>
#include <PdhMsg.h>
#endif

#include "runner.h"

#define PAGE 4096

typedef uint8_t KILOBYTE[1024];
typedef KILOBYTE MEGABYTE[1024];
typedef MEGABYTE GIGABYTE[1024];

#define OVEP_TESTS_DISABLE_COPY(type) type(type const&) = delete;
#define OVEP_TESTS_DISABLE_ASSIGN(type) type& operator=(type const&) = delete;

#define OVEP_TESTS_DISABLE_COPY_AND_ASSIGN(type) \
  OVEP_TESTS_DISABLE_COPY(type)                  \
  OVEP_TESTS_DISABLE_ASSIGN(type)

#define OVEP_TESTS_DEFAULT_MOVE(type) \
  type(type&&) noexcept = default;    \
  type& operator=(type&&) noexcept = default;

#include <nlohmann/json.hpp>

#include "exception.h"
#include "logger.h"
#include "data_utils.h"
#include "timer.h"
#include "network.h"
#include "pipeline.h"
#include "env.h"

#include <onnxruntime/cpu_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/onnxruntime_session_options_config_keys.h>
#include <onnxruntime/onnxruntime_env_config_keys.h>
#include <onnxruntime/onnxruntime_ep_device_ep_metadata_keys.h>

#include <onnxruntime/onnxruntime_experimental_cxx_api.h>

extern Ort::Env env_;

#include "ort_helper.h"
