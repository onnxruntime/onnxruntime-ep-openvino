# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# cmake -P script: embed a manifest file into a PE binary as RT_MANIFEST resource ID 2.
#
# Variables (passed via -D):
#   MT            — absolute path to mt.exe
#   DLL_PATH      — absolute path to the PE file to modify
#   MANIFEST_PATH — absolute path to the .manifest file to embed

cmake_minimum_required(VERSION 3.28)

execute_process(
  COMMAND "${MT}" -nologo
    "-manifest" "${MANIFEST_PATH}"
    "-outputresource:${DLL_PATH};2"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE  _err
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "embed_manifest: '${MT}' failed for '${DLL_PATH}' (exit ${_rc}):\n${_out}${_err}")
endif()
