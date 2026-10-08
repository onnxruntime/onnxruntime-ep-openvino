# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Prebuilt binary subdirectory and glob patterns shared by all packaging targets
# (python wheel, NuGet, MSIX). Defined once here so each subdirectory can reference
# OVEP_PLUGIN_PREBUILT_SUBDIR and OVEP_PLUGIN_INSTALL_GLOB_PATTERNS without re-declaring them.
if(WIN32)
  set(OVEP_PLUGIN_PREBUILT_SUBDIR "bin")
  set(OVEP_PLUGIN_INSTALL_GLOB_PATTERNS
    "onnxruntime_providers_openvino_plugin*.dll"
    "openvino*.dll"
    "tbb*.dll"
    "cache.json"
    "third-party-programs.txt"
    "openvino_runtime.manifest"
  )
else()
  set(OVEP_PLUGIN_PREBUILT_SUBDIR "lib")
  set(OVEP_PLUGIN_INSTALL_GLOB_PATTERNS
    "libonnxruntime_providers_openvino_plugin*.so*"
    "libopenvino*.so*"
    "libtbb*.so*"
    "libhwloc.so*"
    "cache.json"
    "third-party-programs.txt"
  )
endif()

install(FILES "${CMAKE_SOURCE_DIR}/third-party-programs.txt"
  DESTINATION "$<CONFIG>/${OVEP_PLUGIN_PREBUILT_SUBDIR}")

# Python wheel
if(OVEP_PLUGIN_BUILD_PYTHON_WHEEL)
  add_subdirectory(python)
endif()

# NuGet package
if(OVEP_PLUGIN_BUILD_NUGET)
  add_subdirectory(csharp)
endif()

# MSIX loose package (Windows only)
if(WIN32 AND OVEP_PLUGIN_BUILD_MSIX)
  add_subdirectory(winml)
endif()
