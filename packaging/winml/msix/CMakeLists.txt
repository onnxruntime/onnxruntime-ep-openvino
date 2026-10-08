# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Read BUILD_NUMBER from the environment at configure time; default to 9999.
if(DEFINED ENV{BUILD_NUMBER})
  set(MSIX_BUILD_NUMBER "$ENV{BUILD_NUMBER}")
else()
  set(MSIX_BUILD_NUMBER "9999")
  message(STATUS "BUILD_NUMBER not set — MSIX version will use build number 9999")
endif()

# Binary glob patterns are defined in the root CMakeLists.txt as OVEP_PLUGIN_INSTALL_GLOB_PATTERNS.
set(MSIX_GLOB_PATTERNS "${OVEP_PLUGIN_INSTALL_GLOB_PATTERNS}")

set(MSIX_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/AppxManifestTemplate.xml")
set(MSIX_LOGO     "${CMAKE_CURRENT_SOURCE_DIR}/StoreLogo.png")

install(CODE "
  message(STATUS \"Staging loose MSIX package (build number: ${MSIX_BUILD_NUMBER})...\")

  set(msix_dir    \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/winml/msix\")
  set(ep_dir      \"\${msix_dir}/ExecutionProvider\")
  set(images_dir  \"\${msix_dir}/Images\")
  set(_src        \"\${CMAKE_INSTALL_PREFIX}/$<CONFIG>/bin\")

  file(MAKE_DIRECTORY \"\${ep_dir}\")
  file(MAKE_DIRECTORY \"\${images_dir}\")

  set(_patterns \"${MSIX_GLOB_PATTERNS}\")
  foreach(_pattern IN LISTS _patterns)
    file(GLOB _matched \"\${_src}/\${_pattern}\")
    foreach(_f IN LISTS _matched)
      file(COPY \"\${_f}\" DESTINATION \"\${ep_dir}\")
      message(STATUS \"  + \${_f}\")
    endforeach()
  endforeach()

  # Generate AppxManifest.xml from template, substituting @BUILD_NUMBER@
  file(READ \"${MSIX_TEMPLATE}\" _manifest)
  string(REPLACE \"@BUILD_NUMBER@\" \"${MSIX_BUILD_NUMBER}\" _manifest \"\${_manifest}\")
  file(WRITE \"\${msix_dir}/AppxManifest.xml\" \"\${_manifest}\")

  file(COPY \"${MSIX_LOGO}\" DESTINATION \"\${images_dir}\")

  message(STATUS \"Loose package ready. To register:\")
  message(STATUS \"  Add-AppxPackage -Register \\\"\${msix_dir}/AppxManifest.xml\\\"\")
")

message(STATUS "Loose MSIX package will be staged during install (build number: ${MSIX_BUILD_NUMBER})")
