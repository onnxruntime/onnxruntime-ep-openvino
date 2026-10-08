# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

if(NOT DEFINED ENV{ONNXRUNTIME_ROOT_DIR})
  message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT_DIR environment variable to the root directory of ONNX Runtime build")
endif()

file(TO_CMAKE_PATH "$ENV{ONNXRUNTIME_ROOT_DIR}" ONNXRUNTIME_ROOT_DIR)
message(STATUS "ONNXRUNTIME_ROOT_DIR: ${ONNXRUNTIME_ROOT_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_ROOT_DIR}/lib/cmake/onnxruntime")
find_package(onnxruntime 1.30.0 REQUIRED)

if(onnxruntime_FOUND)
  message(STATUS "Found ONNXRuntime: ${onnxruntime_VERSION}")
endif()
