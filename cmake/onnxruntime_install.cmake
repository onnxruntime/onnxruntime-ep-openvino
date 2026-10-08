# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

if(WIN32)
  install(DIRECTORY ${ONNXRUNTIME_ROOT_DIR}/bin/ DESTINATION "$<CONFIG>/bin"
    FILES_MATCHING PATTERN "onnxruntime*.dll"
    REGEX "onnxruntime_providers_openvino_plugin" EXCLUDE)
else()
  install(DIRECTORY ${ONNXRUNTIME_ROOT_DIR}/lib/ DESTINATION "$<CONFIG>/lib"
    FILES_MATCHING PATTERN "libonnxruntime*.so*"
    REGEX "libonnxruntime_providers_openvino_plugin" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE
    PATTERN "cmake" EXCLUDE)
endif()
