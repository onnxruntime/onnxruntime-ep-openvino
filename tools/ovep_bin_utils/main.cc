// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

/**
 * @file ovep_bin_utils.cc
 * @brief OVEP Binary Utilities - Extract and generate OVEP binary cache files
 *
 * This tool provides utilities for working with OpenVINO EP binary cache files:
 * - Extract: Extract blobs and metadata from existing bin files
 * - Generate: Create bin files from compiled model blobs
 * - Verify: Validate bin file structure and metadata
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <sstream>

#include "ov_bin_format.h"
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using namespace onnxruntime::openvino_ep_plugin::bin_format;

// Simple error handling without OVEP_* macros (standalone tool)
template <typename... Args>
static void ToolEnforce(bool condition, Args&&... args) {
  if (!condition) {
    std::ostringstream oss;
    (oss << ... << args);
    throw std::runtime_error(oss.str());
  }
}

#define TOOL_ENFORCE(cond, ...) ToolEnforce((cond), __VA_ARGS__)

// Helper function to check if value is power of two
static inline bool IsPowerOfTwo(size_t value) {
  return value && ((value & (value - 1)) == 0);
}

// Helper function to align up to next alignment boundary
static inline size_t AlignUp(size_t value, size_t alignment) {
  if (!IsPowerOfTwo(alignment)) {
    throw std::runtime_error("Alignment must be a power of two.");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

// Check if offset + size <= total_size without overflow
static inline bool IsWithinBounds(size_t offset, size_t size, size_t total_size) {
  return (offset < total_size) && (size <= total_size) && (offset <= total_size - size);
}

enum class Command {
  Extract,
  Generate,
  Help
};

struct CliArgs {
  Command command;

  // Extract command
  fs::path input_bin;
  fs::path output_dir;

  // Generate command
  std::vector<std::pair<fs::path, std::string>> blobs;  // (path, name) pairs
  fs::path output_bin;
  fs::path metadata_json;  // Optional metadata.json from extract
};

void PrintUsage(const char* program_name) {
  std::cout << "OVEP Bin Utils - Utilities for OVEP binary cache files\n\n";
  std::cout << "Commands:\n\n";
  std::cout << "  Extract blobs from bin file:\n";
  std::cout << "    " << program_name << " extract <input.bin> [--out <output_directory>]\n\n";
  std::cout << "  Generate bin file from blobs:\n";
  std::cout << "    " << program_name << " generate --blob <file> --name <name> --output <output.bin>\n";
  std::cout << "    " << program_name << " generate --blob <file1> --name <name1> --blob <file2> --name <name2> --output <out.bin>\n";
  std::cout << "    " << program_name << " generate --metadata <bin_utils.json> --output <output.bin>\n\n";
  std::cout << "Options:\n";
  std::cout << "  --out <directory>    Output directory for extract (default: <input_stem>_extract/)\n";
  std::cout << "  --blob <file>        Path to compiled model blob file\n";
  std::cout << "  --name <name>        Name for the blob (e.g., partition_0)\n";
  std::cout << "  --metadata <file>    Path to bin_utils.json (from extract command)\n";
  std::cout << "  --output <file>      Output bin file path\n";
  std::cout << "  --help, -h           Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << program_name << " extract model.bin\n";
  std::cout << "  " << program_name << " generate --blob model.blob --name partition_0 --output model.bin\n";
  std::cout << "  " << program_name << " generate --metadata bin_utils.json --output model.bin\n";
}

CliArgs ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    throw std::runtime_error("Error: Missing command");
  }

  CliArgs args;
  std::string command_str = argv[1];

  if (command_str == "--help" || command_str == "-h" || command_str == "help") {
    args.command = Command::Help;
    return args;
  } else if (command_str == "extract") {
    args.command = Command::Extract;

    if (argc < 3) {
      throw std::runtime_error("Error: extract command requires input file argument");
    }

    args.input_bin = argv[2];

    // Parse optional --out argument
    for (int i = 3; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--out") {
        if (i + 1 >= argc) {
          throw std::runtime_error("Error: --out requires a directory path argument");
        }
        args.output_dir = argv[i + 1];
        i++;
      } else {
        throw std::runtime_error("Error: Unknown argument: " + arg);
      }
    }

    // Validate input file exists
    TOOL_ENFORCE(fs::exists(args.input_bin), "Error: Input file does not exist: ", args.input_bin.string());
    TOOL_ENFORCE(fs::is_regular_file(args.input_bin), "Error: Input path is not a file: ", args.input_bin.string());

  } else if (command_str == "generate") {
    args.command = Command::Generate;

    // Parse blob/name pairs and output
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--blob") {
        if (i + 1 >= argc) {
          throw std::runtime_error("Error: --blob requires a file path");
        }
        fs::path blob_path = argv[++i];

        // Next argument should be --name
        if (i + 1 >= argc || std::string(argv[i + 1]) != "--name") {
          throw std::runtime_error("Error: --blob must be followed by --name");
        }
        ++i;  // skip --name

        if (i + 1 >= argc) {
          throw std::runtime_error("Error: --name requires a name value");
        }
        std::string blob_name = argv[++i];

        args.blobs.push_back({blob_path, blob_name});
      } else if (arg == "--output") {
        if (i + 1 >= argc) {
          throw std::runtime_error("Error: --output requires a file path");
        }
        args.output_bin = argv[++i];
      } else if (arg == "--metadata") {
        if (i + 1 >= argc) {
          throw std::runtime_error("Error: --metadata requires a file path");
        }
        args.metadata_json = argv[++i];
      } else {
        throw std::runtime_error("Error: Unknown argument: " + arg);
      }
    }

    if (args.blobs.empty() && args.metadata_json.empty()) {
      throw std::runtime_error("Error: No blobs or metadata specified. Use --blob/--name or --metadata options.");
    }

    if (!args.blobs.empty() && !args.metadata_json.empty()) {
      throw std::runtime_error("Error: Cannot use both --blob/--name and --metadata options together.");
    }

    if (!args.metadata_json.empty()) {
      TOOL_ENFORCE(fs::exists(args.metadata_json), "Error: Metadata file does not exist: ", args.metadata_json.string());
      TOOL_ENFORCE(fs::is_regular_file(args.metadata_json), "Error: Metadata path is not a file: ", args.metadata_json.string());
    }

    if (args.output_bin.empty()) {
      throw std::runtime_error("Error: No output file specified. Use --output option.");
    }

  } else {
    throw std::runtime_error("Error: Unknown command: " + command_str + ". Use extract, generate, or help.");
  }

  return args;
}

fs::path DetermineOutputDirectory(const CliArgs& args) {
  fs::path output_dir;

  if (!args.output_dir.empty()) {
    // User specified --out directory
    TOOL_ENFORCE(fs::exists(args.output_dir), "Error: Specified output directory does not exist: ", args.output_dir.string());
    TOOL_ENFORCE(fs::is_directory(args.output_dir), "Error: Specified output path is not a directory: ", args.output_dir.string());
    output_dir = args.output_dir;
  } else {
    // Create default subdirectory next to input file
    output_dir = args.input_bin.parent_path() / (args.input_bin.stem().string() + "_extract");

    // Create directory if it doesn't exist
    if (!fs::exists(output_dir)) {
      fs::create_directories(output_dir);
      std::cout << "Created output directory: " << output_dir << "\n";
    }
  }

  return output_dir;
}

void ValidateBlobName(const std::string& blob_name) {
  // Prevent path traversal attacks - blob names should not contain path separators
  TOOL_ENFORCE(blob_name.find('/') == std::string::npos && blob_name.find('\\') == std::string::npos,
               "Error: Invalid blob name contains path separator: ", blob_name);
  TOOL_ENFORCE(!blob_name.empty(), "Error: Empty blob name not allowed");
  TOOL_ENFORCE(blob_name != "." && blob_name != "..", "Error: Invalid blob name: ", blob_name);
}

void CheckFileNotExists(const fs::path& file_path) {
  TOOL_ENFORCE(!fs::exists(file_path), "Error: Output file already exists: ", file_path.string(),
               "\nPlease remove existing files or choose a different output directory.");
}

std::vector<uint8_t> ReadBinaryFile(const fs::path& file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  TOOL_ENFORCE(file.is_open(), "Error: Failed to open input file: ", file_path.string());

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  TOOL_ENFORCE(file.good() || file.eof(),
               "Error: Failed to read input file: ", file_path.string());

  return buffer;
}

void ExtractBin(const CliArgs& args) {
  std::cout << "Reading binary file: " << args.input_bin << "\n";

  // Read entire file into memory
  std::vector<uint8_t> data = ReadBinaryFile(args.input_bin);
  TOOL_ENFORCE(!data.empty(), "Error: Input file is empty");

  // Read and validate header
  header_t header{};
  TOOL_ENFORCE(data.size() >= sizeof(header), "Error: File too small for header (", data.size(), " bytes)");
  std::memcpy(&header, data.data(), sizeof(header));

  TOOL_ENFORCE(header.magic == kMagicNumber,
               "Error: Invalid magic number. Expected: 0x", std::hex, kMagicNumber,
               " Got: 0x", header.magic,
               "\nThis does not appear to be a valid OVEP binary file.");

  TOOL_ENFORCE(header.version == to_underlying(BinVersion::v1),
               "Error: Unsupported file version: ", header.version,
               " (expected: ", to_underlying(BinVersion::v1), ")");

  TOOL_ENFORCE(header.header_size == sizeof(header_t),
               "Error: Header size mismatch. Expected: ", sizeof(header_t),
               " Got: ", header.header_size);

  std::cout << "Header validated successfully\n";
  std::cout << "  Version: " << header.version << "\n";
  std::cout << "  BSON offset: " << header.bson_start_offset << "\n";
  std::cout << "  BSON size: " << header.bson_size << " bytes\n";

  // Parse BSON metadata
  TOOL_ENFORCE(IsWithinBounds(header.bson_start_offset, header.bson_size, data.size()),
               "Error: BSON metadata offset/size out of bounds");

  std::vector<uint8_t> bson_data(data.begin() + header.bson_start_offset,
                                 data.begin() + header.bson_start_offset + header.bson_size);
  nlohmann::json metadata = nlohmann::json::from_bson(bson_data);

  TOOL_ENFORCE(metadata.contains(BSONFields::kVersion), "Error: Missing version in BSON metadata");
  TOOL_ENFORCE(metadata.contains(BSONFields::kBlobMetadata), "Error: Missing blob_metadata_map in BSON");

  std::cout << "BSON metadata parsed successfully\n";
  std::cout << "  Schema version: " << metadata[BSONFields::kVersion].get<std::string>() << "\n";

  if (metadata.contains(BSONFields::kProducer)) {
    std::cout << "  Producer: " << metadata[BSONFields::kProducer].get<std::string>() << "\n";
  }

  const auto& blob_map = metadata[BSONFields::kBlobMetadata];
  TOOL_ENFORCE(blob_map.is_object(), "Error: blob_metadata_map must be a JSON object");

  std::cout << "  Blob count: " << blob_map.size() << "\n";

  // Determine output directory
  fs::path output_dir = DetermineOutputDirectory(args);
  std::cout << "\nOutput directory: " << output_dir << "\n";

  // Pre-validate all blob names and check for existing files
  std::cout << "\nValidating blob names and checking for conflicts...\n";
  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    ValidateBlobName(blob_name);

    fs::path blob_file = output_dir / (blob_name + ".blob");
    CheckFileNotExists(blob_file);
  }

  // Check metadata.json doesn't exist
  fs::path metadata_file = output_dir / "metadata.json";
  CheckFileNotExists(metadata_file);

  std::cout << "All validation checks passed\n";

  // Extract blobs
  std::cout << "\nExtracting blobs...\n";
  size_t blob_count = 0;
  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    const uint64_t blob_offset = blob_entry[BSONFields::kDataOffset].get<uint64_t>();
    const uint64_t blob_size = blob_entry[BSONFields::kSize].get<uint64_t>();

    TOOL_ENFORCE(IsWithinBounds(blob_offset, blob_size, data.size()),
                 "Error: Blob '", blob_name, "' offset/size out of bounds");

    fs::path blob_file = output_dir / (blob_name + ".blob");

    std::ofstream out(blob_file, std::ios::binary);
    TOOL_ENFORCE(out.is_open(), "Error: Failed to create blob file: ", blob_file.string());

    out.write(reinterpret_cast<const char*>(data.data() + blob_offset), blob_size);
    TOOL_ENFORCE(out.good(), "Error: Failed to write blob file: ", blob_file.string());
    out.close();

    std::cout << "  [" << (++blob_count) << "/" << blob_map.size() << "] "
              << blob_name << ".blob (" << blob_size << " bytes)\n";
  }

  // Write full metadata JSON (for reference)
  std::cout << "\nWriting metadata.json...\n";
  std::ofstream meta_out(metadata_file);
  TOOL_ENFORCE(meta_out.is_open(), "Error: Failed to create metadata.json: ", metadata_file.string());

  meta_out << metadata.dump(2);  // Pretty-print with 2-space indent
  TOOL_ENFORCE(meta_out.good(), "Error: Failed to write metadata.json");
  meta_out.close();

  // Write simplified bin_utils.json for regeneration
  std::cout << "Writing bin_utils.json...\n";
  fs::path blobs_file = output_dir / "bin_utils.json";
  std::ofstream blobs_out(blobs_file);
  TOOL_ENFORCE(blobs_out.is_open(), "Error: Failed to create bin_utils.json: ", blobs_file.string());

  nlohmann::json blobs_json;

  // Add simplified weights metadata with just locations
  if (metadata.contains(BSONFields::kWeightsMetadata)) {
    const auto& weights_map = metadata[BSONFields::kWeightsMetadata];
    blobs_json[BSONFields::kWeightsMetadata] = nlohmann::json::object();

    for (const auto& [weight_name, weight_entry] : weights_map.items()) {
      nlohmann::json weight_info;

      // Preserve location, data_offset, and size
      if (weight_entry.contains("location")) {
        weight_info["location"] = weight_entry["location"];
      }
      if (weight_entry.contains("data_offset")) {
        weight_info["data_offset"] = weight_entry["data_offset"];
      }
      if (weight_entry.contains("size")) {
        weight_info["size"] = weight_entry["size"];
      }

      blobs_json[BSONFields::kWeightsMetadata][weight_name] = weight_info;
    }
  }

  // Add simplified blob metadata with just locations
  blobs_json[BSONFields::kBlobMetadata] = nlohmann::json::object();

  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    nlohmann::json blob_info;
    blob_info["location"] = blob_name + ".blob";  // Relative path
    blobs_json[BSONFields::kBlobMetadata][blob_name] = blob_info;
  }

  blobs_out << blobs_json.dump(2);  // Pretty-print with 2-space indent
  TOOL_ENFORCE(blobs_out.good(), "Error: Failed to write bin_utils.json");
  blobs_out.close();

  std::cout << "\nExtraction complete!\n";
  std::cout << "  Blobs extracted: " << blob_count << "\n";
  std::cout << "  Output directory: " << output_dir << "\n";
}

// ============================================================================
// Generate Command
// ============================================================================

struct BlobInfo {
  std::string name;
  fs::path path;
  std::vector<uint8_t> data;
};

void LoadBlobsFromMetadata(const fs::path& metadata_path, std::vector<BlobInfo>& blobs, nlohmann::json& weights_metadata) {
  // Read bin_utils.json
  std::ifstream meta_file(metadata_path);
  TOOL_ENFORCE(meta_file.is_open(), "Error: Failed to open blobs file: ", metadata_path.string());

  nlohmann::json metadata;
  meta_file >> metadata;
  meta_file.close();

  TOOL_ENFORCE(metadata.contains(BSONFields::kBlobMetadata), "Error: Missing blob_metadata_map in bin_utils.json");

  const auto& blob_map = metadata[BSONFields::kBlobMetadata];
  TOOL_ENFORCE(blob_map.is_object(), "Error: blob_metadata_map must be a JSON object");

  // Extract weights metadata if present
  if (metadata.contains(BSONFields::kWeightsMetadata)) {
    weights_metadata = metadata[BSONFields::kWeightsMetadata];
  }

  // Get directory containing bin_utils.json
  fs::path metadata_dir = metadata_path.parent_path();

  std::cout << "Loading blobs from bin_utils.json...\n";
  std::cout << "  Directory: " << metadata_dir << "\n";
  std::cout << "  Blobs found: " << blob_map.size() << "\n\n";

  // Load each blob
  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    TOOL_ENFORCE(blob_entry.contains("location"), "Error: Missing 'location' field for blob: ", blob_name);

    std::string location = blob_entry["location"].get<std::string>();
    fs::path blob_file = metadata_dir / location;

    TOOL_ENFORCE(fs::exists(blob_file), "Error: Blob file not found: ", blob_file.string());

    BlobInfo blob;
    blob.name = blob_name;
    blob.path = blob_file;
    blob.data = ReadBinaryFile(blob_file);

    std::cout << "  [OK] " << blob_name << ": " << (blob.data.size() / 1024.0) << " KB\n";
    blobs.push_back(std::move(blob));
  }
}

void GenerateBin(const CliArgs& args) {
  std::vector<BlobInfo> blobs;
  nlohmann::json weights_metadata = nlohmann::json::object();

  // Load blobs either from metadata.json or from command line arguments
  if (!args.metadata_json.empty()) {
    LoadBlobsFromMetadata(args.metadata_json, blobs, weights_metadata);
  } else {
    std::cout << "Loading blobs...\n";

    // Load all blobs from command line
    for (const auto& [blob_path, blob_name] : args.blobs) {
      TOOL_ENFORCE(fs::exists(blob_path), "Error: Blob file not found: ", blob_path.string());

      BlobInfo blob;
      blob.name = blob_name;
      blob.path = blob_path;
      blob.data = ReadBinaryFile(blob_path);

      std::cout << "  [OK] " << blob_name << ": " << (blob.data.size() / 1024.0) << " KB\n";
      blobs.push_back(std::move(blob));
    }
  }

  TOOL_ENFORCE(!blobs.empty(), "Error: No blobs loaded");

  std::cout << "\nGenerating bin file: " << args.output_bin << "\n";
  std::cout << "Blobs: " << blobs.size() << "\n";

  std::ofstream stream(args.output_bin, std::ios::out | std::ios::binary);
  TOOL_ENFORCE(stream.is_open(), "Error: Failed to open output file: ", args.output_bin.string());

  const auto stream_start = stream.tellp();

  // Write alignment padding
  auto write_alignment_padding = [&stream](uint64_t current_pos, uint64_t alignment) {
    uint64_t aligned_position = AlignUp(current_pos, alignment);
    uint64_t padding_size = aligned_position - current_pos;
    if (padding_size > 0) {
      std::vector<char> padding(padding_size, 0);
      stream.write(padding.data(), padding.size());
      TOOL_ENFORCE(stream.good(), "Error: Failed to write alignment padding");
    }
  };

  // Reserve space for header (will be updated later)
  header_t header{};
  header.magic = kMagicNumber;
  header.version = to_underlying(BinVersion::current);
  header.header_size = sizeof(header_t);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  TOOL_ENFORCE(stream.good(), "Error: Failed to write header");

  // Build JSON metadata
  nlohmann::json j;
  j[BSONFields::kVersion] = BSONFields::kCurrentBsonVersion;
  j[BSONFields::kProducer] = "ovep_bin_utils";

  // Add weights metadata (preserve data_offset and size from original if available)
  nlohmann::json full_weights_metadata = nlohmann::json::object();
  if (weights_metadata.is_object() && !weights_metadata.empty()) {
    for (const auto& [weight_name, weight_entry] : weights_metadata.items()) {
      nlohmann::json full_entry;

      // Copy location if present
      if (weight_entry.contains("location")) {
        full_entry["location"] = weight_entry["location"];
      }

      // Use preserved data_offset and size if available, otherwise default to 0
      if (weight_entry.contains("data_offset")) {
        full_entry["data_offset"] = weight_entry["data_offset"];
      } else {
        full_entry["data_offset"] = 0;
      }

      if (weight_entry.contains("size")) {
        full_entry["size"] = weight_entry["size"];
      } else {
        full_entry["size"] = 0;
      }

      full_weights_metadata[weight_name] = full_entry;
    }
  }
  j[BSONFields::kWeightsMetadata] = full_weights_metadata;

  // Add blob metadata with placeholder values
  nlohmann::json blob_map = nlohmann::json::object();
  for (const auto& blob : blobs) {
    nlohmann::json blob_entry;
    auto max_val = std::numeric_limits<int64_t>::max();
    blob_entry[BSONFields::kDataOffset] = max_val;
    blob_entry[BSONFields::kSize] = max_val;
    blob_map[blob.name] = blob_entry;
  }
  j[BSONFields::kBlobMetadata] = blob_map;

  // Write BSON metadata (will be rewritten later with correct blob info)
  header.bson_start_offset = stream.tellp();

  size_t orig_bson_size;
  {
    std::vector<uint8_t> bson_data = nlohmann::json::to_bson(j);
    orig_bson_size = bson_data.size();
    stream.write(reinterpret_cast<const char*>(bson_data.data()), bson_data.size());
    TOOL_ENFORCE(stream.good(), "Error: Failed to write BSON data");
  }
  uint64_t bson_end = stream.tellp();

  write_alignment_padding(bson_end, kBlobAlignment);

  // Write blob data and capture actual offsets/sizes
  for (auto& blob : blobs) {
    uint64_t blob_start = stream.tellp();
    stream.write(reinterpret_cast<const char*>(blob.data.data()), blob.data.size());
    TOOL_ENFORCE(stream.good(), "Error: Failed to write blob data for: ", blob.name);

    uint64_t blob_end = stream.tellp();
    uint64_t blob_size = blob_end - blob_start;

    // Update blob metadata in BSON
    auto& blob_entry = j[BSONFields::kBlobMetadata][blob.name];
    blob_entry[BSONFields::kDataOffset] = blob_start;
    blob_entry[BSONFields::kSize] = blob_size;

    std::cout << "  " << blob.name << ": offset=" << blob_start
              << ", size=" << blob_size << " bytes\n";

    write_alignment_padding(blob_end, kBlobAlignment);
  }

  // Rewrite BSON metadata with correct blob info
  std::vector<uint8_t> updated_bson_data = nlohmann::json::to_bson(j);
  TOOL_ENFORCE(updated_bson_data.size() <= orig_bson_size,
               "Error: BSON size larger after updating blob info. Original: ", orig_bson_size,
               " Updated: ", updated_bson_data.size());

  stream.seekp(header.bson_start_offset);
  stream.write(reinterpret_cast<const char*>(updated_bson_data.data()), updated_bson_data.size());
  TOOL_ENFORCE(stream.good(), "Error: Failed to rewrite BSON data");
  bson_end = stream.tellp();
  header.bson_size = bson_end - header.bson_start_offset;

  // Update header with BSON offsets
  stream.seekp(stream_start);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  TOOL_ENFORCE(stream.good(), "Error: Failed to update header");

  stream.seekp(0, std::ios::end);
  auto file_size = stream.tellp();
  stream.close();

  TOOL_ENFORCE(!stream.fail(), "Error occurred while writing to file");

  std::cout << "\n[SUCCESS] Bin file generated successfully\n";
  std::cout << "  Size: " << (file_size / (1024.0 * 1024.0)) << " MB\n";
}

int main(int argc, char* argv[]) {
  try {
    CliArgs args = ParseArgs(argc, argv);

    switch (args.command) {
      case Command::Help:
        PrintUsage(argv[0]);
        return 0;

      case Command::Extract:
        ExtractBin(args);
        return 0;

      case Command::Generate:
        GenerateBin(args);
        return 0;

      default:
        std::cerr << "Error: Unknown command\n";
        return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "\n"
              << e.what() << "\n\n";
    return 1;
  } catch (...) {
    std::cerr << "\nUnknown error occurred\n\n";
    return 1;
  }
}
