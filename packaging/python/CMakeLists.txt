# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Check if Python 3 is available
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Found Python: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

# Check if build package is available
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m build --version
  RESULT_VARIABLE build_check
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT build_check EQUAL 0)
  message(FATAL_ERROR "OVEP_PLUGIN_BUILD_PYTHON_WHEEL enabled but 'build' package not found. Install it with: ${Python3_EXECUTABLE} -m pip install build")
endif()

# Binary glob patterns and subdirectory are defined in the root CMakeLists.txt
# as OVEP_PLUGIN_INSTALL_GLOB_PATTERNS and OVEP_PLUGIN_PREBUILT_SUBDIR.
set(PREBUILT_SUBDIR    "${OVEP_PLUGIN_PREBUILT_SUBDIR}")
set(PY_GLOB_PATTERNS   "${OVEP_PLUGIN_INSTALL_GLOB_PATTERNS}")

# Compute wheel version string at configure time.
# OVEP_PLUGIN_BUILD_METADATA (".<type>.<number>", set in the root CMakeLists.txt)
# is appended to the PEP 440 local-version segment for non-production builds so
# wheels sharing a git commit remain distinguishable; empty for local/dev builds.
if(OVEP_PLUGIN_PRODUCTION_BUILD)
  set(PY_WHEEL_VERSION "${OVEP_VERSION_STRING}")
elseif(GIT_COMMIT_ID)
  set(PY_WHEEL_VERSION "${OVEP_VERSION_STRING}.dev0+${GIT_COMMIT_ID}${OVEP_PLUGIN_BUILD_METADATA}")
elseif(OVEP_PLUGIN_BUILD_METADATA)
  # No commit but build metadata present: strip the leading '.' to start the
  # PEP 440 local segment (e.g. "+ci.4567").
  string(REGEX REPLACE "^\\." "" _py_local "${OVEP_PLUGIN_BUILD_METADATA}")
  set(PY_WHEEL_VERSION "${OVEP_VERSION_STRING}.dev0+${_py_local}")
else()
  set(PY_WHEEL_VERSION "${OVEP_VERSION_STRING}.dev0")
endif()

set(PY_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_ep_openvino")

# Generate pyproject.toml from template, embedding the wheel version
configure_file(pyproject.toml.in "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" @ONLY)

install(CODE "
  message(STATUS \"Building Python wheel package...\")

  # Stage pre-built binaries into the package source directory
  set(prebuilt_dir \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/${PREBUILT_SUBDIR}\")
  message(STATUS \"Staging binaries from: \${prebuilt_dir}\")
  foreach(pattern IN ITEMS ${PY_GLOB_PATTERNS})
    file(GLOB matched_files \"\${prebuilt_dir}/\${pattern}\")
    foreach(f IN LISTS matched_files)
      file(COPY \"\${f}\" DESTINATION \"${PY_PKG_DIR}\")
    endforeach()
  endforeach()

  # Clean stale Python build artifacts to avoid mixing Debug/Release binaries
  file(REMOVE_RECURSE \"${CMAKE_CURRENT_SOURCE_DIR}/build\")

  execute_process(
    COMMAND \"${Python3_EXECUTABLE}\" -m build --wheel --no-isolation --outdir \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/python/dist\"
    WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\"
    RESULT_VARIABLE result
  )

  # Clean up staged files from the source tree
  foreach(pattern IN ITEMS ${PY_GLOB_PATTERNS})
    file(GLOB staged_files \"${PY_PKG_DIR}/\${pattern}\")
    foreach(f IN LISTS staged_files)
      file(REMOVE \"\${f}\")
    endforeach()
  endforeach()

  if(result)
    message(FATAL_ERROR \"Failed to build Python wheel (exit code: \${result})\")
  else()
    message(STATUS \"Python wheel created in \${CMAKE_INSTALL_PREFIX}/$<CONFIG>/python/dist/\")
  endif()
")

message(STATUS "Python wheel will be built during install")
