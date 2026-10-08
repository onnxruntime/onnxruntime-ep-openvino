// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ovep_tests {

// Convert an fs::path to a UTF-8 encoded std::string.
// On Windows, fs::path::string() uses the ANSI code page (CP_ACP) and will
// throw or corrupt characters outside that page.  u8string() always yields
// the correct UTF-8 representation of the path regardless of platform.
inline std::string path_to_utf8(const fs::path& p) {
  auto u8 = p.u8string();
  return {u8.begin(), u8.end()};
}

// Construct an fs::path from a UTF-8 encoded std::string.
// On Windows, the fs::path(std::string) constructor uses CP_ACP, not UTF-8.
// Using the u8string constructor preserves non-ASCII characters correctly.
inline fs::path path_from_utf8(std::string_view utf8) {
  return fs::path{std::u8string(utf8.begin(), utf8.end())};
}

// Overload for u8 string literals (const char8_t*) and std::u8string_view.
inline fs::path path_from_utf8(std::u8string_view utf8) {
  return fs::path{std::u8string(utf8)};
}

// Non-member operator+ for fs::path concatenation via UTF-8.
// The standard only provides member operator+= (concat); there is no non-member
// operator+(path, string). This overload enables expressions such as:
//   stem + "_ctx_" + device + ".onnx"
// All appended strings must be valid UTF-8 (ASCII suffixes always qualify).
inline fs::path operator+(const fs::path& lhs, std::string_view rhs) {
  return path_from_utf8(path_to_utf8(lhs) + std::string(rhs));
}

}  // namespace ovep_tests
