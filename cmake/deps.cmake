# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Add dependencies
include(FetchContent)

function(group_targets_by_folder SOURCE_DIR FOLDER_NAME)
  get_property(targets DIRECTORY ${SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target ${targets})
      if(TARGET ${target})
          set_target_properties(${target} PROPERTIES FOLDER ${FOLDER_NAME})
      endif()
  endforeach()
endfunction()

# Fetched dependencies below are static; enable PIC so they link into the shared plugin/test targets.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add GSL
FetchContent_Declare(
  gsl
  GIT_REPOSITORY  https://github.com/microsoft/GSL.git
  GIT_TAG         v4.0.0
  GIT_SHALLOW     ON
)
FetchContent_MakeAvailable(gsl)

# Add nlohmann_json
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY  https://github.com/nlohmann/json.git
  GIT_TAG         v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# Add googletests
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0  # Use a specific tag or commit
  EXCLUDE_FROM_ALL
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
