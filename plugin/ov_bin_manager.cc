// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#include "ov_bin_manager.h"
#include "ov_bin_format.h"
#include "ov_large_string_buf.h"
#include "ov_shared_context.h"
#include "ov_plugin_utils.h"
#include "ov_plugin_ver.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <span>

namespace onnxruntime {
namespace openvino_ep_plugin {

using namespace bin_format;

void BinManager::AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model) {
  std::unique_lock lock(mutex_);
  auto [it, inserted] = native_blobs_.try_emplace(name, BlobContainer{.compiled_model = compiled_model, .tensor = {}});
  OVEP_ENFORCE(inserted, "Blob already exists: ", name);
}

const ov::Tensor BinManager::GetNativeBlob(const std::string& blob_name) const {
  std::shared_lock lock(mutex_);

  auto it = native_blobs_.find(blob_name);
  OVEP_ENFORCE(it != native_blobs_.end(), "Blob not found for ", blob_name);

  auto& blob_container = it->second;
  OVEP_ENFORCE(blob_container.tensor, "Blob tensor data is not available for ", blob_name);
  return blob_container.tensor;
}

void BinManager::SetWeightsFile(const std::filesystem::path& weights_file) {
  std::unique_lock lock(mutex_);
  OVEP_ENFORCE(weights_file_.empty() || weights_file_ == weights_file,
               "Error: The bin records a single external weights file. Already set to ",
               ToUtf8String(weights_file_), ", got ", ToUtf8String(weights_file));
  weights_file_ = weights_file;
}

std::filesystem::path BinManager::GetWeightsFile() const {
  std::shared_lock lock(mutex_);
  return weights_file_;
}

void BinManager::Serialize(std::shared_ptr<SharedContext> shared_context) {
  auto path = GetExternalBinPath();
  std::ofstream stream(path, std::ios::out | std::ios::binary);
  OVEP_ENFORCE(stream.is_open(), "Failed to open file for serialization: ", ToUtf8String(path));
  Serialize(stream, shared_context);
}

std::string BinManager::SerializeToString(std::shared_ptr<SharedContext> shared_context) {
  detail::LargeStringBuf buffer;
  std::ostream stream(&buffer);
  Serialize(stream, std::move(shared_context));
  OVEP_ENFORCE(stream.good(), "Failed to serialize binary data.");
  return std::move(buffer).TakeString();
}

void BinManager::Serialize(std::ostream& stream, std::shared_ptr<SharedContext> shared_context) {
  std::shared_lock ul(mutex_);

  auto metadata = shared_context ? shared_context->GetMetadataCopy() : SharedContext::Metadata::Map{};
  if (metadata.empty() && native_blobs_.empty()) {
    return;  // Nothing to serialize
  }

  const auto stream_start = stream.tellp();

  auto write_alignment_padding = [&stream](uint64_t current_pos, uint64_t alignment) {
    uint64_t aligned_position = AlignUp(current_pos, alignment);
    uint64_t padding_size = aligned_position - current_pos;
    if (padding_size > 0) {
      std::vector<char> padding(padding_size, 0);
      stream.write(padding.data(), padding.size());
      OVEP_ENFORCE(stream.good(), "Error: Failed to write alignment padding.");
    }
  };

  // Reserve space for header (will be updated later)
  header_t header{};
  header.magic = kMagicNumber;
  header.version = to_underlying(BinVersion::current);
  header.header_size = sizeof(header_t);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  OVEP_ENFORCE(stream.good(), "Error: Failed to write header.");

  // Build JSON metadata
  nlohmann::json j;
  j[BSONFields::kVersion] = BSONFields::kCurrentBsonVersion;
  j[BSONFields::kProducer] = "onnxruntime-openvino-ep-plugin-" OVEP_PLUGIN_VER_TAGGED;

  // Written only for weightless exports, so its presence means the blobs need an external weights file
  if (!weights_file_.empty()) {
    j[BSONFields::kWeightsFile] = ToUtf8String(weights_file_);
  }

  // Add weights metadata as a map (from SharedContext if available)
  if (!metadata.empty()) {
    nlohmann::json weights_map = nlohmann::json::object();
    for (const auto& [key, value] : metadata) {
      nlohmann::json weight_entry;
      weight_entry[BSONFields::kLocation] = ToUtf8String(value.serialized.location);
      weight_entry[BSONFields::kDataOffset] = value.serialized.data_offset;
      weight_entry[BSONFields::kSize] = value.serialized.size;
      weights_map[key] = weight_entry;
    }
    j[BSONFields::kWeightsMetadata] = weights_map;
  }

  // Add blob metadata with placeholder values as a map (will be updated after writing blobs)
  nlohmann::json blob_map = nlohmann::json::object();
  for (const auto& [key, value] : native_blobs_) {
    nlohmann::json blob_entry;
    auto max_val = std::numeric_limits<int64_t>::max();
    // Placehold max size since we don't know actual offsets/sizes yet, and if they aren't max they might serialize smaller.
    blob_entry[BSONFields::kDataOffset] = max_val;
    blob_entry[BSONFields::kSize] = max_val;
    blob_map[key] = blob_entry;
  }
  j[BSONFields::kBlobMetadata] = blob_map;

  // Write BSON metadata (will be rewritten later with correct blob info)
  header.bson_start_offset = stream.tellp();

  size_t orig_bson_size;
  {
    std::vector<uint8_t> bson_data = nlohmann::json::to_bson(j);
    orig_bson_size = bson_data.size();
    stream.write(reinterpret_cast<const char*>(bson_data.data()), bson_data.size());
    OVEP_ENFORCE(stream.good(), "Error: Failed to write BSON data.");
  }
  uint64_t bson_end = stream.tellp();

  write_alignment_padding(bson_end, kBlobAlignment);

  // Write blob data and capture actual offsets/sizes
  for (auto& [blob_name, value] : native_blobs_) {
    uint64_t blob_start = stream.tellp();
    value.compiled_model.export_model(stream);
    OVEP_ENFORCE(stream.good(), "Error: Failed to write blob data for ", blob_name);
    // Seek to end of stream after writing in case export model didn't leave us there
    stream.seekp(0, std::ios::end);
    uint64_t blob_end = stream.tellp();
    uint64_t blob_size = blob_end - blob_start;

    // Update blob metadata in BSON
    auto& blob_entry = j[BSONFields::kBlobMetadata][blob_name];
    blob_entry[BSONFields::kDataOffset] = blob_start;
    blob_entry[BSONFields::kSize] = blob_size;

    write_alignment_padding(blob_end, kBlobAlignment);
  }

  // Rewrite BSON metadata with correct blob info
  std::vector<uint8_t> updated_bson_data = nlohmann::json::to_bson(j);
  OVEP_ENFORCE(updated_bson_data.size() <= orig_bson_size,
               "Error: BSON size larger after updating blob info. Original: ", orig_bson_size,
               " Updated: ", updated_bson_data.size());

  stream.seekp(header.bson_start_offset);
  stream.write(reinterpret_cast<const char*>(updated_bson_data.data()), updated_bson_data.size());
  OVEP_ENFORCE(stream.good(), "Error: Failed to rewrite BSON data.");
  bson_end = stream.tellp();
  header.bson_size = bson_end - header.bson_start_offset;

  // Update header with BSON offsets
  stream.seekp(stream_start);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  OVEP_ENFORCE(stream.good(), "Error: Failed to update header.");

  stream.seekp(0, std::ios::end);  // Move to end after writing.
}

void BinManager::Deserialize(std::shared_ptr<SharedContext> shared_context) {
  {
    std::shared_lock lock(mutex_);
    if (is_deserialized_) {
      return;  // Already deserialized; skip.
    }
  }

  auto path = GetExternalBinPath();
  OVEP_ENFORCE(!path.empty(), "External bin path not set.");

  try {
    // Memory-map the file using OpenVINO; route through the OwnedBuffer path so the
    // mapped tensor's lifetime is managed the same way as host-allocator-backed buffers.
    auto mapped = std::make_shared<ov::Tensor>(ov::read_tensor_data(path));
    OVEP_ENFORCE(*mapped, "Failed to memory-map external bin file: ", ToUtf8String(path));

    const auto& const_mapped = static_cast<const ov::Tensor&>(*mapped);
    OwnedBuffer buffer{
        .data = std::span<const uint8_t>(const_mapped.data<const uint8_t>(), const_mapped.get_byte_size()),
        .owner = std::shared_ptr<void>(mapped, mapped.get()),
    };
    Deserialize(std::move(buffer), shared_context);
  } catch (const std::exception& e) {
    OVEP_THROW(e.what(), "\nCould not deserialize binary data. This could mean the bin is corrupted or incompatible. Try re-generating ep context cache.");
  }
}

void BinManager::Deserialize(OwnedBuffer buffer, std::shared_ptr<SharedContext> shared_context) {
  try {
    std::unique_lock lock(mutex_);

    if (is_deserialized_) {
      return;  // Already deserialized; skip.
    }

    OVEP_ENFORCE(buffer_owner_ == nullptr,
                 "BinManager::Deserialize: buffer is already populated. "
                 "Overwriting it would invalidate tensors that reference the old buffer.");

    // Take ownership of the caller-supplied bytes and publish a non-owning ov::Tensor view
    // over them. Blob ROI tensors slice into mapped_bin_; buffer_owner_ keeps the underlying
    // bytes alive for as long as any of those slices are reachable.
    buffer_owner_ = std::move(buffer.owner);
    auto* data_ptr = const_cast<uint8_t*>(buffer.data.data());
    mapped_bin_ = ov::Tensor(ov::element::u8, ov::Shape{buffer.data.size()}, data_ptr);

    DeserializeImpl(buffer.data, shared_context);
  } catch (const std::exception& e) {
    // Release ownership of the buffer to avoid dangling references in case of failure
    buffer_owner_.reset();
    mapped_bin_ = ov::Tensor{};
    OVEP_THROW(e.what(), "\nCould not deserialize binary data. This could mean the bin is corrupted or incompatible. Try re-generating ep context cache.");
  }
}

void BinManager::DeserializeImpl(std::span<const uint8_t> data, const std::shared_ptr<SharedContext>& shared_context) {
  OVEP_ENFORCE(!data.empty(), "Error: Cannot deserialize from empty buffer.");

  // Fetch a required field from a BSON sub-object. The blob metadata may originate from an
  // untrusted or corrupted source; a missing key must be a clean, catchable error rather than
  // undefined behavior. nlohmann's const operator[] is UB when the key is absent, so validate
  // presence explicitly here.
  auto require_field = [](const nlohmann::json& obj, const char* key) -> const nlohmann::json& {
    OVEP_ENFORCE(obj.is_object() && obj.contains(key), "Error: Missing required BSON field: ", key);
    return obj[key];
  };

  // Read and validate header
  header_t header{};
  OVEP_ENFORCE(data.size() >= sizeof(header), "Error: Buffer too small for header.");
  std::memcpy(&header, data.data(), sizeof(header));

  OVEP_ENFORCE(header.magic == kMagicNumber, "Error: Invalid magic number. Expected: 0x", std::hex, kMagicNumber, " Got: 0x", header.magic);
  OVEP_ENFORCE(header.version == to_underlying(BinVersion::current), "Error: Unsupported file version: ", header.version);
  OVEP_ENFORCE(header.header_size == sizeof(header_t), "Error: Header size mismatch.");

  // Parse BSON metadata
  OVEP_ENFORCE(IsWithinBounds(header.bson_start_offset, header.bson_size, data.size()),
               "Error: BSON start offset + size out of bounds.");
  nlohmann::json j;
  j = nlohmann::json::from_bson(data.subspan(header.bson_start_offset, header.bson_size));

  // Validate BSON version (check major version compatibility)
  OVEP_ENFORCE(j.contains(BSONFields::kVersion), "Error: Missing version in BSON metadata.");
  auto bson_version = j[BSONFields::kVersion].get<std::string>();

  // Extract major version from semver strings (format: "major.minor.patch")
  auto get_major_version = [](const std::string& version) -> int {
    size_t dot_pos = version.find('.');
    if (dot_pos == std::string::npos) return -1;
    try {
      return std::stoi(version.substr(0, dot_pos));
    } catch (...) {
      return -1;
    }
  };

  int file_major = get_major_version(bson_version);
  int current_major = get_major_version(BSONFields::kCurrentBsonVersion);

  OVEP_ENFORCE(file_major >= 0 && current_major >= 0,
               "Error: Invalid BSON version format. Expected: ", BSONFields::kCurrentBsonVersion,
               " Got: ", bson_version);
  OVEP_ENFORCE(file_major == current_major,
               "Error: Incompatible BSON schema major version. Expected: ", current_major,
               " Got: ", file_major, " (full version: ", bson_version, ")");

  if (j.contains(BSONFields::kWeightsFile)) {
    auto weights_file = j[BSONFields::kWeightsFile].get<std::string>();
    weights_file_ = PathFromUtf8(weights_file);
    OVEP_ENFORCE(IsSafeRelativeSubpath(weights_file_),
                 "Error: Weights file must be a relative path within the model directory: ", weights_file);
  }

  // Parse weights metadata and populate SharedContext if available
  if (j.contains(BSONFields::kWeightsMetadata)) {
    OVEP_ENFORCE(shared_context, "Error: Bin contains shared weights metadata but no SharedContext was provided during deserialization.");
    const auto& weights_map = j[BSONFields::kWeightsMetadata];
    if (weights_map.is_object()) {
      for (const auto& [weight_name, weight_entry] : weights_map.items()) {
        auto location = require_field(weight_entry, BSONFields::kLocation).get<std::string>();
        const auto weight_location = PathFromUtf8(location);
        OVEP_ENFORCE(IsSafeRelativeSubpath(weight_location),
                     "Error: External weight location must be a relative path within the model directory: ", location);
        auto data_offset = require_field(weight_entry, BSONFields::kDataOffset).get<size_t>();
        auto size = require_field(weight_entry, BSONFields::kSize).get<size_t>();
        shared_context->AddExternalWeight(weight_name, data_offset, size, weight_location);
      }
    }
  }

  // Parse blob metadata and create zero-copy tensor views
  OVEP_ENFORCE(j.contains(BSONFields::kBlobMetadata), "Error: Missing blob metadata in BSON.");
  const auto& blob_map = j[BSONFields::kBlobMetadata];
  OVEP_ENFORCE(blob_map.is_object(), "Error: Blob metadata must be an object.");

  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    const uint64_t blob_offset = require_field(blob_entry, BSONFields::kDataOffset).get<uint64_t>();
    const uint64_t blob_size = require_field(blob_entry, BSONFields::kSize).get<uint64_t>();

    OVEP_ENFORCE(IsWithinBounds(blob_offset, blob_size, data.size()),
                 "Error: Blob offset + size out of bounds for ", blob_name);

    BlobContainer container{};
    // Create tensor view from mapped_bin_ (which holds the underlying buffer)
    ov::Coordinate begin{blob_offset};
    ov::Coordinate end{blob_offset + blob_size};
    container.tensor = ov::Tensor(mapped_bin_, begin, end);

    const bool inserted = native_blobs_.try_emplace(blob_name, std::move(container)).second;
    OVEP_ENFORCE(inserted, "Duplicate native blob name in EPContext: ", blob_name);
  }

  is_deserialized_ = true;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
