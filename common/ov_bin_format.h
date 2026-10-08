// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace onnxruntime {
namespace openvino_ep_plugin {
namespace bin_format {

/*
    Logical layout of the single binary file:
    [Header]
    [BSON Metadata]              ← Contains blob_metadata_map with data_offset and size for each blob
    [Padding to 64K alignment]   ← Blob section starts here (64K aligned)
    [Blob 1]                     ← BSON blob_metadata_map["blob_name"].data_offset points here
    [Padding to 64K alignment]   ← Each blob end is 64K aligned
    [Blob 2]                     ← BSON blob_metadata_map["blob_name2"].data_offset points here
    [Padding to 64K alignment]
    [Blob 3]                     ← BSON blob_metadata_map["blob_name3"].data_offset points here
    ...

    BSON Schema:
    {
      "version": <string>,                    // BSON schema version (semver format)
      "producer": <string>,                   // Producer identifier (e.g., "onnxruntime-openvino-ep-plugin")
      "weights_file": <string, optional>,     // Model-relative external weights file, present for weightless blobs
      "weights_metadata_map": {               // Map of ONNX tensor names to external weight file metadata
        "<tensor_name>": {
          "location": <string>,               // Relative path to external weights file
          "data_offset": <int64>,            // Offset within external weights file
          "size": <int64>                    // Size of weight data in bytes
        },
        ...
      },
      "blob_metadata_map": {                  // Map of blob names to compiled model blob metadata
        "<blob_name>": {
          "data_offset": <int64>,            // Absolute file offset to blob data (64K aligned)
          "size": <int64>                    // Actual blob data size (excluding padding)
        },
        ...
      }
    }

    Note: data_offset values in blob_metadata_map are absolute file positions.
          size values exclude alignment padding bytes.
*/

// "OVEP_BIN" in little-endian (memory will read as 'O','V','E','P','_','B','I','N')
constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;

enum class BinVersion : uint64_t {
  v1 = 1,
  current = v1
};

struct header_t {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

constexpr uint64_t kBlobAlignment = 64 * 1024;

// BSON field names
namespace BSONFields {
constexpr const char* kVersion = "version";
constexpr const char* kProducer = "producer";
constexpr const char* kWeightsMetadata = "weights_metadata_map";
constexpr const char* kBlobMetadata = "blob_metadata_map";
constexpr const char* kLocation = "location";
constexpr const char* kDataOffset = "data_offset";
constexpr const char* kSize = "size";
constexpr const char* kWeightsFile = "weights_file";
constexpr const char* kCurrentBsonVersion = "1.1.0";
}  // namespace BSONFields

template <typename E>
constexpr std::underlying_type_t<E> to_underlying(E e) noexcept {
  static_assert(std::is_enum_v<E>, "to_underlying requires an enum type");
  return static_cast<std::underlying_type_t<E>>(e);
}

// The overall bin-format compatibility version embedded in the EP compatibility string.
// This value is chosen deliberately (it is NOT derived from the versions below):
//   - major: a breaking change to the on-disk format (header layout or BSON schema). Gated for
//            compatibility — a mismatch rejects the cached model.
//   - minor: an additive, backward-compatible change. Recorded for diagnostics only; not gated.
//
// The static_asserts below pin the underlying versions this value was last reconciled against.
// Bump BinVersion or kCurrentBsonVersion and one of them fails to compile, forcing a deliberate
// review of kBinCompatVersion before the new underlying version can ship.
inline constexpr std::string_view kBinCompatVersion = "1.1";

static_assert(to_underlying(BinVersion::current) == 1,
              "BinVersion changed. Review kBinCompatVersion (a header layout change is breaking -> "
              "bump its major), then update the expected value in this static_assert.");

static_assert(std::string_view(BSONFields::kCurrentBsonVersion) == "1.1.0",
              "kCurrentBsonVersion changed. Review kBinCompatVersion (bump major if the BSON change "
              "is breaking, minor if additive), then update the expected value in this static_assert.");

inline std::string GetBinCompatVersion() { return std::string(kBinCompatVersion); }

}  // namespace bin_format
}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
