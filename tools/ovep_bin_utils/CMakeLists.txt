# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# OVEP Bin Utils Tool - Extract, generate, and verify bin files
add_executable(ovep_bin_utils main.cc)

# Set C++ standard to match project
set_property(TARGET ovep_bin_utils PROPERTY CXX_STANDARD 20)
set_property(TARGET ovep_bin_utils PROPERTY CXX_STANDARD_REQUIRED ON)

# Include common headers for ov_bin_format.h
target_include_directories(ovep_bin_utils PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

# Link only nlohmann_json (no OpenVINO/ORT dependencies needed)
target_link_libraries(ovep_bin_utils PRIVATE
    nlohmann_json)

# Install to bin directory
install(TARGETS ovep_bin_utils RUNTIME DESTINATION "$<CONFIG>/bin")
