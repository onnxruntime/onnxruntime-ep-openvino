// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ortAbiTestFixture.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using namespace ovep_tests;
using namespace ovep_tests::onnxruntime_abi;

using session_options_t = ortAbiTestFixture::session_options_wrapper_t;
using model_library_t = ortAbiTestFixture::model_library_t;

namespace {
// Minimal mirror of common/ov_bin_format.h layout, kept local so the test target need not depend on
// the plugin's private headers. Must stay in sync.
struct bin_header_t {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};
static_assert(sizeof(bin_header_t) == 40, "bin_header_t must match common/ov_bin_format.h header_t");

constexpr const char* kBsonBlobMetadata = "blob_metadata_map";
constexpr const char* kBsonWeightsMetadata = "weights_metadata_map";
constexpr const char* kBsonLocation = "location";
constexpr const char* kBsonDataOffset = "data_offset";
constexpr const char* kBsonSize = "size";

// Which map, and which required field to strip from its first entry.
struct malformed_field_case_t {
  const char* section;
  const char* field;
  const char* label;
};

std::vector<uint8_t> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("read_file: cannot open " + path_to_utf8(path));
  }
  const std::streamsize size = static_cast<std::streamsize>(in.tellg());
  if (size < 0) {
    throw std::runtime_error("read_file: tellg failed for " + path_to_utf8(path));
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

void write_file(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

fs::path find_external_bin(const fs::path& dir) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".bin") {
      return entry.path();
    }
  }
  return {};
}
}  // namespace

///////////////////////////////////////////////////////////////////////
// Security regression: a blob or weight entry missing a required field must fail cleanly, not hit
// const operator[] UB in BinManager::DeserializeImpl. gemm_external with shared EP contexts emits a
// bin carrying both maps, so one generated bin covers every field require_field guards.
struct ort_abi_ep_ctx_malformed_metadata
    : public ortAbiTestFixture,
      public testing::WithParamInterface<malformed_field_case_t> {};

TEST_P(ort_abi_ep_ctx_malformed_metadata, missing_required_field_fails_cleanly) {
  const auto& test_case = GetParam();
  const std::string ov_device_name = "CPU";
  const auto& model_info = model_library_t::GetModel("gemm_external");
  const fs::path epctx_model = temp_dir_ / (std::string("gemm_external_ctx_") + test_case.label + ".onnx");

  // Generate a shared EPContext whose blobs and external weights land in an external .bin.
  {
    session_options_t session_options(
        ov_device_name,
        {{kOrtSessionOptionEpContextEnable, "1"},
         {kOrtSessionOptionShareEpContexts, "1"},
         {kOrtSessionOptionStopShareEpContexts, "1"},
         {kOrtSessionOptionEpContextFilePath, path_to_utf8(epctx_model)},
         {kOrtSessionOptionEpContextEmbedMode, "0"}});
    model_info.create_session(std::move(session_options));
  }
  ASSERT_TRUE(fs::exists(epctx_model));

  const fs::path bin_path = find_external_bin(temp_dir_);
  ASSERT_FALSE(bin_path.empty()) << "external bin not written under " << path_to_utf8(temp_dir_);

  // Strip the field and rewrite the bin. Layout is [header][BSON][blob...]; DeserializeImpl throws at
  // the missing-field check before reaching blob data, so we keep only [header][mutated BSON] and
  // patch the header's bson_size to match.
  std::vector<uint8_t> bin = read_file(bin_path);
  ASSERT_GE(bin.size(), sizeof(bin_header_t));
  bin_header_t header{};
  std::memcpy(&header, bin.data(), sizeof(header));
  ASSERT_LE(header.bson_start_offset + header.bson_size, bin.size());

  auto j = nlohmann::json::from_bson(
      std::vector<uint8_t>(bin.begin() + header.bson_start_offset,
                           bin.begin() + header.bson_start_offset + header.bson_size));

  ASSERT_TRUE(j.contains(test_case.section)) << "test precondition: bin missing " << test_case.section;
  auto& section_map = j[test_case.section];
  ASSERT_TRUE(section_map.is_object() && !section_map.empty()) << "test precondition: empty " << test_case.section;
  section_map.begin().value().erase(test_case.field);

  const auto new_bson = nlohmann::json::to_bson(j);
  std::vector<uint8_t> mutated(bin.begin(), bin.begin() + header.bson_start_offset);
  mutated.insert(mutated.end(), new_bson.begin(), new_bson.end());
  header.bson_size = new_bson.size();
  std::memcpy(mutated.data(), &header, sizeof(header));
  write_file(bin_path, mutated);

  // Loading the EPContext model must surface a clean error (exception), not crash/UB.
  auto epctx_model_info = model_info;
  epctx_model_info.model_path_ = epctx_model;

  std::string error_message;
  try {
    auto session = epctx_model_info.create_session(
        session_options_t(ov_device_name, {{kOrtSessionOptionShareEpContexts, "1"}}));
    FAIL() << "session creation must fail when a required " << test_case.section << " field is missing";
  } catch (const std::exception& e) {
    error_message = e.what();
  }

  EXPECT_NE(error_message.find("BSON field"), std::string::npos)
      << "expected a missing-field error from Deserialize, got: " << error_message;
  EXPECT_NE(error_message.find(test_case.field), std::string::npos)
      << "error should name the missing field '" << test_case.field << "', got: " << error_message;
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_ep_ctx_malformed_metadata,
    ort_abi_ep_ctx_malformed_metadata,
    testing::Values(
        malformed_field_case_t{kBsonBlobMetadata, kBsonDataOffset, "blob_data_offset"},
        malformed_field_case_t{kBsonBlobMetadata, kBsonSize, "blob_size"},
        malformed_field_case_t{kBsonWeightsMetadata, kBsonLocation, "weight_location"},
        malformed_field_case_t{kBsonWeightsMetadata, kBsonDataOffset, "weight_data_offset"},
        malformed_field_case_t{kBsonWeightsMetadata, kBsonSize, "weight_size"}),
    [](const testing::TestParamInfo<malformed_field_case_t>& info) { return info.param.label; });
