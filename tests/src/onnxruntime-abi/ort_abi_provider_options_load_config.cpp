// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

///////////////////////////////////////////////////////////////////////
using device_t = ortAbiTestFixture::device_wrapper_t;
using graph_t = ortAbiTestFixture::graph_wrapper_t;

///////////////////////////////////////////////////////////////////////
struct ort_abi_provider_options_load_config_expect_invalid_argument
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::pair<std::string, std::string> > {};

///////////////////////////////////////////////////////////////////////////
TEST_P(ort_abi_provider_options_load_config_expect_invalid_argument, run) {
  const std::string& load_config_val = GetParam().second;

  device_t device("CPU");
  network_t network{};
  data_utils::read_binary_file("testdata/mul_1.onnx", network.buffer_);

  network.ort_provider_options_["load_config"] = load_config_val;
  try {
    graph_t graph(device, network);

    ADD_FAILURE() << "Expected Ort::Exception, but no exception was thrown";
  } catch (const Ort::Exception& e) {
    // We expect to receive an exception with error code = ORT_INVALID_ARGUMENT (2).
    // There is currently an issue where ORT Core re-routes the error code returned from the EP to ORT_FAIL (1).
    // The error message is preserved, so as a workaround, check message to see if it contains "INVALID_ARGUMENT".
    const std::string ort_error_msg(e.what());
    if (ort_error_msg.find("INVALID_ARGUMENT") == std::string::npos) {
      ADD_FAILURE() << "Expected ORT Exception message to contain 'INVALID_ARGUMENT'. "
                    << "It is: " << ort_error_msg;
    }
  } catch (...) {
    ADD_FAILURE() << "Caught unexpected non-ORT exception.";
  }
}

///////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ort_abi_provider_options_load_config_expect_invalid_argument,
    ort_abi_provider_options_load_config_expect_invalid_argument,
    testing::Values(
        std::make_pair("malformed_json", "{\"CPU\": {\"CACHE_DIR\":"),
        std::make_pair("non_object_at_root", "123"),
        std::make_pair("unsupported_array_type", "{\"CPU\":{\"KEY\":[1,2,3,4,5]}}"),
        std::make_pair("unsupported_null_type", "{\"CPU\":{\"KEY\":null}}"),
        std::make_pair(
            "too_deep_nested_map",
            "{\"CPU\":{\"L1\":{\"L2\":{\"L3\":{\"L4\":{\"L5\":{\"L6\":{\"L7\":{\"L8\":{\"L9\":{}}}}}}}}}}}"),
        std::make_pair("no_device_map", "{\"CACHE_DIR\": \"temp\"}")),
    [](const testing::TestParamInfo<ort_abi_provider_options_load_config_expect_invalid_argument::ParamType>& info) { return info.param.first; });

///////////////////////////////////////////////////////////////////////////
struct ort_abi_provider_options_load_config : public ortAbiTestFixture {};

///////////////////////////////////////////////////////////////////////////
TEST_F(ort_abi_provider_options_load_config, parse_json) {
  device_t device("CPU");
  network_t network{};
  data_utils::read_binary_file("testdata/mul_1.onnx", network.buffer_);

  static const std::string valid_json = R"JSON(
    {
      "AUTO": {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "ENABLE_STARTUP_FALLBACK": "NO",
        "ENABLE_RUNTIME_FALLBACK": "NO",
        "DEVICE_PROPERTIES": {
          "CPU": {
            "INFERENCE_PRECISION_HINT": "f32",
            "NUM_STREAMS": "3",
            "LOG_LEVEL": "LOG_DEBUG"
          },
          "GPU": {
            "NUM_STREAMS": "5",
            "EXECUTION_MODE_HINT": "ACCURACY",
            "LOG_LEVEL": "LOG_DEBUG",
            "PLUGIN_THROTTLE": true
          },
          "NPU": {
            "MAX_PROMPT_LEN": "4096",
            "MIN_RESPONSE_LEN": "128",
            "++PREFILL_CONFIG": {
              "NPUW_DEVICES": "NPU,CPU",
              "NPUW_ONLINE_AVOID": "P:SinCos/NPU"
            },
            "++GENERATE_CONFIG": {
              "NPUW_DEVICES": "NPU,CPU",
              "NPUW_ONLINE_AVOID": "P:SinCos/NPU"
            }
          }
        },
        "LOG_LEVEL": "LOG_DEBUG"
      },
      "NPU": {
        "MAX_PROMPT_LEN": "4096",
        "MIN_RESPONSE_LEN": "128",
        "++PREFILL_CONFIG": {
          "NPUW_DEVICES": "NPU,CPU",
          "NPUW_ONLINE_AVOID": "P:SinCos/NPU"
        },
        "++GENERATE_CONFIG": {
          "NPUW_DEVICES": "NPU,CPU",
          "NPUW_ONLINE_AVOID": "P:SinCos/NPU"
        }
      }
    }
    )JSON";

  network.ort_provider_options_["load_config"] = valid_json;
  graph_t graph(device, network);
}