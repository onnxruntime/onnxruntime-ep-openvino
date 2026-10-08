# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

find_program(COVSELECT_EXECUTABLE covselect REQUIRED)

set(COV_FILE_ARG "")
if(DEFINED ENV{COVFILE})
  set(COV_FILE_ARG "-f" "$ENV{COVFILE}")
endif()

execute_process(
  COMMAND ${COVSELECT_EXECUTABLE} ${COV_FILE_ARG} --deleteAll
  RESULT_VARIABLE _covselect_result)
if(NOT _covselect_result EQUAL 0)
  message(FATAL_ERROR "covselect --deleteAll failed (exit code ${_covselect_result})")
endif()

file(GLOB_RECURSE _excluded_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.h")

foreach(_src ${_excluded_sources})
  cmake_path(NATIVE_PATH _src _src_native)
  execute_process(
    COMMAND ${COVSELECT_EXECUTABLE} ${COV_FILE_ARG} --add "!${_src_native}"
    RESULT_VARIABLE _covselect_result)
  if(NOT _covselect_result EQUAL 0)
    message(FATAL_ERROR "covselect --add failed for ${_src_native} (exit code ${_covselect_result})")
  endif()
endforeach()
