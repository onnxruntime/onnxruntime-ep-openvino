# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME "ovep_tests")

project(${TARGET_NAME})

# Check if git-lfs is installed
find_program(GIT_LFS_EXECUTABLE git-lfs)

if(NOT GIT_LFS_EXECUTABLE)
    message(WARNING "git-lfs is not installed. Test data files may not be properly downloaded. "
        "Please install git-lfs and run 'git lfs install' and 'git lfs pull'.")
endif()

include(./../cmake/deps.cmake)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Add executable
add_executable(
    ${TARGET_NAME}
    ${SOURCES})

target_precompile_headers(${TARGET_NAME} PRIVATE include/pch.h)

target_include_directories(
    ${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fixtures
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime-abi
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin
    ${ONNXRUNTIME_ROOT_DIR}/include)

target_link_directories(
    ${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/lib)

target_link_libraries(
    ${TARGET_NAME}
    GTest::gmock
    onnxruntime
    nlohmann_json)

if(MSVC)
    target_link_libraries(
        ${TARGET_NAME}
        Pdh
        winmm
        d3d12
        dxgi)
endif()

# Set additional target properties
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 23)

if(MSVC)
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "/ignore:4099 /DEBUG")
endif()

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(${TARGET_NAME} Threads::Threads)
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH FALSE)
endif()

# ovep_tests doesn't link against OpenVINO, so OPENVINO_VERSION_MAJOR/MINOR from
# openvino/core/version.hpp is unavailable; inject the version from CMake instead.
target_compile_definitions(
    ${TARGET_NAME} PRIVATE
    OVEP_OV_VERSION_MAJOR=${OpenVINO_VERSION_MAJOR}
    OVEP_OV_VERSION_MINOR=${OpenVINO_VERSION_MINOR})

# Needed for MSVC to correctly set the value of __cplusplus macro
if(MSVC)
    target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        "/W4"
        "/WX"
        "/external:anglebrackets"
        "/external:W0"
        "/Zc:__cplusplus"
        "/bigobj"
        "/MP")
else()
    target_compile_options(${TARGET_NAME} PRIVATE "-Wall" "-Wextra" "-Werror")
endif()

include(GoogleTest)

# Use an imported target so CTest runs the installed executable rather than the
# build-tree target, whose runtime dependencies are not staged together.
set(OVEP_INSTALLED_TEST_TARGET "${TARGET_NAME}_installed")
add_executable(${OVEP_INSTALLED_TEST_TARGET} IMPORTED)
set(OVEP_INSTALLED_TEST_FILENAME "${TARGET_NAME}${CMAKE_EXECUTABLE_SUFFIX}")

if(CMAKE_CONFIGURATION_TYPES)
    set(OVEP_INSTALLED_TEST_CONFIGURATIONS ${CMAKE_CONFIGURATION_TYPES})
else()
    set(OVEP_INSTALLED_TEST_CONFIGURATIONS "${CMAKE_BUILD_TYPE}")
endif()

set_target_properties(
    ${OVEP_INSTALLED_TEST_TARGET}
    PROPERTIES
    IMPORTED_CONFIGURATIONS "${OVEP_INSTALLED_TEST_CONFIGURATIONS}")

foreach(config IN LISTS OVEP_INSTALLED_TEST_CONFIGURATIONS)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(
        ${OVEP_INSTALLED_TEST_TARGET}
        PROPERTIES
        "IMPORTED_LOCATION_${config_upper}"
        "${CMAKE_INSTALL_PREFIX}/${config}/bin/${OVEP_INSTALLED_TEST_FILENAME}")
endforeach()

gtest_discover_tests(
    ${OVEP_INSTALLED_TEST_TARGET}
    WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/$<CONFIG>/bin"
    DISCOVERY_MODE PRE_TEST)

# Set up visual studio filters
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${SOURCES})

source_group("Generated Files" FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/cmake_pch.cxx)

foreach(config_type ${CMAKE_CONFIGURATION_TYPES})
    source_group("Generated Files" FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/${config_type}/cmake_pch.hxx)
endforeach()

# Install step
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION $<CONFIG>/bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testdata/ DESTINATION $<CONFIG>/bin/testdata PATTERN "*.gitattributes" EXCLUDE)
