# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Install OpenVINO runtime and TBB binaries.
#
# Windows: installs from <build>/sxs_staging/<config>/ — files produced by
#   cmake/openvino.cmake at build time with Authenticode signatures stripped
#   and SxS dep manifests embedded.  OVEP_OV_TBB_STAGED_FILES_* are set there.
#
# Linux: installs directly from the OpenVINO source tree.  File lists are
#   enumerated at configure time by openvino.cmake and stored in
#   OVEP_OV_TBB_INSTALL_FILES_{Release,Debug,RelWithDebInfo} or OVEP_OV_TBB_INSTALL_FILES_wheel.

if(WIN32)
  set(_bin_subdir "bin")
  if(DEFINED OVEP_OV_TBB_STAGED_FILES_wheel)
    install(FILES ${OVEP_OV_TBB_STAGED_FILES_wheel} DESTINATION "$<CONFIG>/${_bin_subdir}")
    install(FILES "${OVEP_ASSEMBLY_MANIFEST_wheel}"
      RENAME "openvino_runtime.manifest"
      DESTINATION "$<CONFIG>/${_bin_subdir}")
  else()
    foreach(_config IN ITEMS Release Debug RelWithDebInfo)
      if(OVEP_OV_TBB_STAGED_FILES_${_config})
        install(FILES ${OVEP_OV_TBB_STAGED_FILES_${_config}}
          DESTINATION "$<CONFIG>/${_bin_subdir}"
          CONFIGURATIONS ${_config})
      endif()
      if(OVEP_ASSEMBLY_MANIFEST_${_config})
        install(FILES "${OVEP_ASSEMBLY_MANIFEST_${_config}}"
          RENAME "openvino_runtime.manifest"
          DESTINATION "$<CONFIG>/${_bin_subdir}"
          CONFIGURATIONS ${_config})
      endif()
    endforeach()
  endif()
else()
  set(_bin_subdir "lib")
  if(DEFINED OVEP_OV_TBB_INSTALL_FILES_wheel)
    install(FILES ${OVEP_OV_TBB_INSTALL_FILES_wheel} DESTINATION "$<CONFIG>/${_bin_subdir}")
  else()
    foreach(_config IN ITEMS Release Debug RelWithDebInfo)
      if(OVEP_OV_TBB_INSTALL_FILES_${_config})
        install(FILES ${OVEP_OV_TBB_INSTALL_FILES_${_config}}
          DESTINATION "$<CONFIG>/${_bin_subdir}"
          CONFIGURATIONS ${_config})
      endif()
    endforeach()
  endif()
endif()
