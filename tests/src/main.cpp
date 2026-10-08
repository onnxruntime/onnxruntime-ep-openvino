// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

///////////////////////////////////////////////////////////////////////
using namespace ovep_tests;
using testing::GTEST_FLAG(repeat);

///////////////////////////////////////////////////////////////////////
static OrtLoggingLevel get_ort_log_level() {
  auto custom_log = get_environment_variable("OVEP_TESTS_ORT_LOG_LEVEL");
  if (custom_log.has_value()) {
    if (*custom_log == "VERBOSE")
      return ORT_LOGGING_LEVEL_VERBOSE;
    else if (*custom_log == "INFO")
      return ORT_LOGGING_LEVEL_INFO;
    else if (*custom_log == "WARNING")
      return ORT_LOGGING_LEVEL_WARNING;
    else if (*custom_log == "ERROR")
      return ORT_LOGGING_LEVEL_ERROR;
    else if (*custom_log == "FATAL")
      return ORT_LOGGING_LEVEL_FATAL;
  }
  return ORT_LOGGING_LEVEL_WARNING;  // Default log level
}

///////////////////////////////////////////////////////////////////////
// Lets a fresh ovep_tests.exe process exercise env-level EP config (e.g.
// ep.openvinoexecutionprovider.* keys, or ORT's own "allow_virtual_devices") read via
// Ort::GetEnvConfigEntries(), which can only be set on the first OrtEnv created in a process --
// run in isolation via:
//   set OVEP_TESTS_HARDWARE_DEVICE_TYPE_FILTER=NPU && ovep_tests.exe --gtest_filter=...
static Ort::Env MakeEnv() {
  Ort::KeyValuePairs config_entries{};
  if (auto filter = get_environment_variable("OVEP_TESTS_HARDWARE_DEVICE_TYPE_FILTER")) {
    config_entries.Add("ep.openvinoexecutionprovider.hardware_device_type", filter->c_str());
  }
  if (get_environment_variable("OVEP_TESTS_ALLOW_VIRTUAL_DEVICES").has_value()) {
    config_entries.Add("allow_virtual_devices", "1");
  }

  OrtEnvCreationOptions options{};
  options.version = ORT_API_VERSION;
  options.logging_severity_level = get_ort_log_level();
  options.log_id = "ovep_tests";
  options.config_entries = config_entries;
  return Ort::Env(&options);
}

Ort::Env env_ = MakeEnv();

///////////////////////////////////////////////////////////////////////
GTEST_API_ int real_main(int argc, char** argv) {
  // Initialize google test framework and run all tests
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  try {
    return real_main(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Unhandled unknown exception" << std::endl;
    return EXIT_FAILURE;
  }
}
