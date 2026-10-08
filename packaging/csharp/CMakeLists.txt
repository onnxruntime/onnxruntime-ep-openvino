# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Binary glob patterns and subdirectory are defined in the root CMakeLists.txt
# as OVEP_PLUGIN_INSTALL_GLOB_PATTERNS and OVEP_PLUGIN_PREBUILT_SUBDIR.
if(WIN32)
  set(NUGET_RID "win-x64")
else()
  set(NUGET_RID "linux-x64")
endif()
set(NUGET_PREBUILT_SUBDIR "${OVEP_PLUGIN_PREBUILT_SUBDIR}")
set(NUGET_GLOB_PATTERNS   "${OVEP_PLUGIN_INSTALL_GLOB_PATTERNS}")

find_program(DOTNET_EXECUTABLE dotnet)
if(NOT DOTNET_EXECUTABLE)
  message(FATAL_ERROR "OVEP_PLUGIN_BUILD_NUGET enabled but 'dotnet' not found. Install the .NET SDK.")
endif()
message(STATUS "Found dotnet: ${DOTNET_EXECUTABLE}")

set(NUGET_CSPROJ    "${CMAKE_CURRENT_SOURCE_DIR}/Intel.ML.OnnxRuntime.EP.OpenVINO/Intel.ML.OnnxRuntime.EP.OpenVINO.csproj")
set(NUGET_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Intel.ML.OnnxRuntime.EP.OpenVINO/runtimes/${NUGET_RID}/native")

# Derive production flag from OVEP_PLUGIN_PRODUCTION_BUILD (set via -DOVEP_PLUGIN_PRODUCTION_BUILD=ON)
if(OVEP_PLUGIN_PRODUCTION_BUILD)
  set(OVEP_IS_PRODUCTION "true")
else()
  set(OVEP_IS_PRODUCTION "false")
endif()

install(CODE "
  message(STATUS \"Staging native files for NuGet from \${CMAKE_INSTALL_PREFIX}/$<CONFIG>/${NUGET_PREBUILT_SUBDIR}\")
  file(REMOVE_RECURSE \"${NUGET_NATIVE_DIR}\")
  file(MAKE_DIRECTORY  \"${NUGET_NATIVE_DIR}\")
  set(_src \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/${NUGET_PREBUILT_SUBDIR}\")
  set(_patterns \"${NUGET_GLOB_PATTERNS}\")
  foreach(_pattern IN LISTS _patterns)
    file(GLOB _matched \"\${_src}/\${_pattern}\")
    foreach(_f IN LISTS _matched)
      file(COPY \"\${_f}\" DESTINATION \"${NUGET_NATIVE_DIR}\")
      message(STATUS \"  + \${_f}\")
    endforeach()
  endforeach()

  # Optional env vars: OVEP_PLUGIN_NUGET_SOURCES (semicolon-separated list of extra package sources)
  #                    OVEP_PLUGIN_NUGET_PACKAGES (cache directory)
  set(_restore_args restore \"${NUGET_CSPROJ}\")
  if(DEFINED ENV{OVEP_PLUGIN_NUGET_SOURCES})
    set(_nuget_sources \"\$ENV{OVEP_PLUGIN_NUGET_SOURCES}\")
    foreach(_src IN LISTS _nuget_sources)
      list(APPEND _restore_args --source \"\${_src}\")
    endforeach()
  endif()
  if(DEFINED ENV{OVEP_PLUGIN_NUGET_PACKAGES})
    list(APPEND _restore_args --packages \"\$ENV{OVEP_PLUGIN_NUGET_PACKAGES}\")
  endif()

  message(STATUS \"Restoring NuGet dependencies...\")
  execute_process(
    COMMAND \"${DOTNET_EXECUTABLE}\" \${_restore_args}      -p:MSBuildEnableWorkloadResolver=false
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR \"Failed to restore NuGet dependencies (exit code: \${result})\")
  endif()

  message(STATUS \"Building C# assembly...\")
  execute_process(
    COMMAND \"${DOTNET_EXECUTABLE}\" build \"${NUGET_CSPROJ}\" -c Release --no-restore
           -p:MSBuildEnableWorkloadResolver=false -p:GeneratePackageOnBuild=false
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR \"Failed to build C# assembly (exit code: \${result})\")
  endif()

  message(STATUS \"Packing NuGet package...\")
  execute_process(
    COMMAND \"${DOTNET_EXECUTABLE}\" pack \"${NUGET_CSPROJ}\" -c Release --no-restore --no-build
      --output \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/csharp/nuget\"      -p:MSBuildEnableWorkloadResolver=false
      -p:OvepIsProduction=${OVEP_IS_PRODUCTION} -p:OvepGitCommitId=${GIT_COMMIT_ID} -p:OvepBuildMetadata=${OVEP_PLUGIN_BUILD_METADATA}
    RESULT_VARIABLE result
  )
  if(result)
    message(FATAL_ERROR \"Failed to build NuGet package (exit code: \${result})\")
  else()
    message(STATUS \"NuGet package created in \${CMAKE_INSTALL_PREFIX}/$<CONFIG>/csharp/nuget\")
  endif()
")

message(STATUS "NuGet package will be built during install")
