# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.28...4.0)

# Read project version from the single source-of-truth VERSION_NUMBER file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION_NUMBER" OVEP_VERSION_STRING)
string(STRIP "${OVEP_VERSION_STRING}" OVEP_VERSION_STRING)

if(NOT OVEP_VERSION_STRING MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
  message(FATAL_ERROR
    "VERSION_NUMBER must contain a version of the form MAJOR.MINOR.PATCH.\n"
    "Got: \"${OVEP_VERSION_STRING}\"")
endif()

set(OVEP_PLUGIN_MAJOR_VERSION "${CMAKE_MATCH_1}")
set(OVEP_PLUGIN_MINOR_VERSION "${CMAKE_MATCH_2}")
set(OVEP_PLUGIN_PATCH_VERSION "${CMAKE_MATCH_3}")

project(OpenVINOPluginEP VERSION "${OVEP_VERSION_STRING}" LANGUAGES CXX)

# Option for production release builds — controls version suffix and PRODUCTION macro
option(OVEP_PLUGIN_PRODUCTION_BUILD "Production release build (disables -dev version suffix)" OFF)

# Generate version header from template — output goes to the build tree
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/common/ov_plugin_ver.h.in"
  "${CMAKE_BINARY_DIR}/common/ov_plugin_ver.h"
  @ONLY)

# Allow version.rc (#include "common/ov_plugin_ver.h") to resolve the generated header
if(WIN32)
  string(APPEND CMAKE_RC_FLAGS " /I\"${CMAKE_BINARY_DIR}\"")
endif()

include(cmake/onnxruntime.cmake)

if(WIN32)
  # Locate a tool in the newest available Windows SDK x64 bin directory.
  # Usage: ovep_find_winsdk_tool(<cache_var> <exe_name>)
  function(ovep_find_winsdk_tool out_var exe_name)
    if(${out_var})
      return()
    endif()

    set(_hints "")

    if(DEFINED ENV{WDKBinRoot})
      list(APPEND _hints "$ENV{WDKBinRoot}/x64")
    endif()

    file(GLOB _glob_hints
      "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64"
      "C:/Program Files/Windows Kits/10/bin/10.*/x64")
    list(SORT _glob_hints COMPARE NATURAL ORDER DESCENDING)
    list(APPEND _hints ${_glob_hints})
    find_program(${out_var} "${exe_name}" HINTS ${_hints} NO_DEFAULT_PATH)

    if(NOT ${out_var})
      message(FATAL_ERROR
        "${exe_name}.exe not found in known Windows SDK paths.\n"
        "Set -D${out_var}=<path/to/${exe_name}.exe> to override.")
    endif()

    message(STATUS "Found ${exe_name}.exe: ${${out_var}}")
  endfunction()

  ovep_find_winsdk_tool(MT mt)
  ovep_find_winsdk_tool(SIGNTOOL signtool)
endif()

include(cmake/openvino.cmake)

# Get the current git commit ID
if("${GIT_COMMIT_ID}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse --short HEAD
    OUTPUT_VARIABLE GIT_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Optional CI build metadata appended to version strings so builds sharing a git
# commit remain distinguishable. Supplied by the out-of-tree CI, e.g.
# -DOVEP_PLUGIN_BUILD_TYPE=ci -DOVEP_PLUGIN_BUILD_NUMBER=4567
# Left empty for local/dev builds, in which case the version string is unchanged.
set(OVEP_PLUGIN_BUILD_TYPE "" CACHE STRING "CI build type (ci|verify|prod); appended to version build metadata")
set(OVEP_PLUGIN_BUILD_NUMBER "" CACHE STRING "CI build number; appended to version build metadata")

# Assemble a dot-separated SemVer build-metadata suffix ".<type>.<number>"
# (either part optional). Sanitised to the SemVer build-metadata charset [0-9A-Za-z-].
set(OVEP_PLUGIN_BUILD_METADATA "")

foreach(_part IN ITEMS "${OVEP_PLUGIN_BUILD_TYPE}" "${OVEP_PLUGIN_BUILD_NUMBER}")
  if(_part)
    string(REGEX REPLACE "[^0-9A-Za-z-]" "-" _part "${_part}")
    string(APPEND OVEP_PLUGIN_BUILD_METADATA ".${_part}")
  endif()
endforeach()

# Option to enable MSVC static runtime
option(OVEP_PLUGIN_USE_MSVC_STATIC_RUNTIME "Use static MSVC runtime library" ON)

# Option to build Python wheel during install
option(OVEP_PLUGIN_BUILD_PYTHON_WHEEL "Build Python wheel package during install" OFF)

# Option to build NuGet package during install
option(OVEP_PLUGIN_BUILD_NUGET "Build NuGet package during install" OFF)

# Option to build MSIX package during install (Windows only)
if(WIN32)
  option(OVEP_PLUGIN_BUILD_MSIX "Build MSIX package during install" OFF)
endif()

# Option to skip building tests
option(OVEP_PLUGIN_BUILD_TESTS "Build tests" ON)

# Option to skip building tools
option(OVEP_PLUGIN_BUILD_TOOLS "Build tools" ON)

# Option to enable Bullseye code coverage instrumentation
option(OVEP_PLUGIN_ENABLE_BULLSEYE_COVERAGE "Enable Bullseye code coverage instrumentation" OFF)

# Set MSVC runtime library
if(MSVC)
  if(OVEP_PLUGIN_USE_MSVC_STATIC_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  endif()
endif()

# Windows: prevent windows.h from defining min/max macros that clash with std::min/std::max.
# Set here (before add_subdirectory) so it applies uniformly to all targets: plugin, tests, tools, packaging.
if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()

include(cmake/openvino_install.cmake)
include(cmake/onnxruntime_install.cmake)

add_subdirectory(plugin)

if(OVEP_PLUGIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(OVEP_PLUGIN_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

add_subdirectory(packaging)

if(OVEP_PLUGIN_ENABLE_BULLSEYE_COVERAGE)
  # COVFILE must be set in the shell environment before invoking CMake and the build tool.
  if(NOT DEFINED ENV{COVFILE})
    message(FATAL_ERROR "COVFILE environment variable must be set when enabling Bullseye coverage.")
  endif()

  file(TO_CMAKE_PATH "$ENV{COVFILE}" COVFILE_PATH)
  include(cmake/bullseye_excludes.cmake)
  install(FILES "${COVFILE_PATH}" DESTINATION "$<CONFIG>/bin")
endif()
