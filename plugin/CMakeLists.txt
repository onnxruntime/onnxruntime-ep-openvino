# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

set(TARGET_NAME onnxruntime_providers_openvino_plugin)

# Version variables are set in the root CMakeLists.txt from VERSION_NUMBER
include(./../cmake/deps.cmake)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime/*.h
  ${INTEL_OPENVINO_DIR}/runtime/include/openvino/*.hpp)

source_group("Header Files\\onnxruntime" REGULAR_EXPRESSION "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime/.*\\.h")
source_group("Header Files\\openvino" REGULAR_EXPRESSION "${INTEL_OPENVINO_DIR}/runtime/include/openvino/.*\\.hpp")

add_library(${TARGET_NAME} SHARED ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)

if(UNIX)
  set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${OVEP_PLUGIN_MAJOR_VERSION}.${OVEP_PLUGIN_MINOR_VERSION}.${OVEP_PLUGIN_PATCH_VERSION}"
    SOVERSION "${OVEP_PLUGIN_MAJOR_VERSION}")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 23)

target_compile_definitions(${TARGET_NAME} PRIVATE
  GIT_COMMIT_ID="${GIT_COMMIT_ID}"
  OVEP_PLUGIN_BUILD_METADATA="${OVEP_PLUGIN_BUILD_METADATA}"
  FILE_NAME=\"${TARGET_NAME}.dll\")

if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE
    "/W4"
    "/WX"
    "/external:anglebrackets"
    "/external:W0"
    "/guard:cf"
    "/Qspectre"
    "/sdl"
    "/Zi")
  target_link_options(${TARGET_NAME} PRIVATE
    "/DEBUG"
    "/INCREMENTAL:NO"
    "/OPT:REF"
    "/OPT:ICF"
    "/guard:cf"
    "/DYNAMICBASE"
    "/CETCOMPAT"
    "-DEF:${CMAKE_CURRENT_SOURCE_DIR}/symbols.def")
else()
  target_compile_options(${TARGET_NAME} PRIVATE "-Wall" "-Wextra" "-Werror")
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_BINARY_DIR}/common
    ${ONNXRUNTIME_ROOT_DIR}/include)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::frontend::onnx
    Microsoft.GSL::GSL
    nlohmann_json)

# Set RPATH for Linux to find libraries in the same directory
if(UNIX)
  set_target_properties(${TARGET_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH_USE_ORIGIN TRUE
    BUILD_WITH_INSTALL_RPATH FALSE)
  message(STATUS "RPATH configured for ${TARGET_NAME}: $ORIGIN")
endif()

if(OVEP_PLUGIN_PRODUCTION_BUILD)
  target_compile_definitions(${TARGET_NAME} PRIVATE PRODUCTION)
endif()

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION "$<CONFIG>/bin" LIBRARY DESTINATION "$<CONFIG>/lib")

if(WIN32)
  # Pre-generate the SxS dep manifest for this DLL at configure time.
  set(_manifest_dir "${CMAKE_BINARY_DIR}/sxs_manifests")
  file(MAKE_DIRECTORY "${_manifest_dir}")
  set(DLL_BASE_NAME   "${TARGET_NAME}")
  set(EP_FILE_VERSION "${OVEP_VERSION_STRING}.0")
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/sxs/dep.manifest.in"
    "${_manifest_dir}/${TARGET_NAME}.dep.manifest"
  )

  # Post-link: embed the SxS dep manifest into the plugin DLL.
  set(_manifest "${_manifest_dir}/${TARGET_NAME}.dep.manifest")
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
      "-DMT=${MT}"
      "-DDLL_PATH=$<TARGET_FILE:${TARGET_NAME}>"
      "-DMANIFEST_PATH=${_manifest}"
      -P "${CMAKE_SOURCE_DIR}/cmake/sxs/embed_manifest.cmake"
    COMMENT "SxS: embedding dep manifest into ${TARGET_NAME}.dll"
    VERBATIM
  )
endif()
