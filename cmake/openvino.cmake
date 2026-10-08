# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Locate OpenVINO: honour INTEL_OPENVINO_DIR (set by setupvars.bat/sh) when present;
# otherwise fall back to an installed openvino Python wheel.
if(DEFINED ENV{INTEL_OPENVINO_DIR})
  file(TO_CMAKE_PATH "$ENV{INTEL_OPENVINO_DIR}" INTEL_OPENVINO_DIR)
  message(STATUS "OpenVINO: using archive/installer at ${INTEL_OPENVINO_DIR}")
else()
  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import openvino, pathlib; print(pathlib.Path(openvino.__file__).parent)"
      OUTPUT_VARIABLE _OVEP_OV_WHEEL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _OVEP_OV_WHEEL_RESULT
    )
  endif()
  if(Python3_FOUND AND _OVEP_OV_WHEEL_RESULT EQUAL 0 AND _OVEP_OV_WHEEL_DIR)
    file(TO_CMAKE_PATH "${_OVEP_OV_WHEEL_DIR}" INTEL_OPENVINO_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_OPENVINO_DIR}/cmake")
    message(STATUS "OpenVINO: found Python wheel at ${INTEL_OPENVINO_DIR}")
  else()
    message(FATAL_ERROR
      "OpenVINO not found.  Either:\n"
      "  1. Run setupvars.bat / setupvars.sh before configuring, or\n"
      "  2. Install the openvino Python wheel:  pip install openvino")
  endif()
endif()

find_package(OpenVINO 2026.4 REQUIRED COMPONENTS Runtime ONNX)
message(STATUS "Found OpenVINO: ${OpenVINO_VERSION}")

# ---------------------------------------------------------------------------
# Enumerate OV and TBB binaries for install and SxS manifest embedding.
#
# Sets (archive layout, Windows):
#   OVEP_OV_TBB_INSTALL_FILES_Release        — full source paths for Release config
#   OVEP_OV_TBB_INSTALL_FILES_Debug          — full source paths for Debug config
#   OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo — full source paths for RelWithDebInfo config
#
# Sets (archive layout, Linux — no per-config subdirs, same files for all configs):
#   OVEP_OV_TBB_INSTALL_FILES_Release        )
#   OVEP_OV_TBB_INSTALL_FILES_Debug          )  all set to the same file list
#   OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo )
#
# Sets (wheel layout, both platforms):
#   OVEP_OV_TBB_INSTALL_FILES_wheel          — full source paths (config-independent)
#
# Sets (Windows only, both layouts):
#   OVEP_OV_TBB_DLL_NAMES             — DLL filenames for SxS dep manifest embedding
# ---------------------------------------------------------------------------

# Glob patterns selecting the OV binaries needed by OVEP.
# Applied as "<dir>/<pattern>" — works for both Windows (.dll) and Linux (.so*).
set(_OVEP_OV_PATTERNS
  "cache.json"
  "*openvino.*"
  "*openvinod.*"
  "*openvino*plugin*"
  "*openvino*compiler*"
  "*openvino*npu*runtime*"
  "*openvino_ir_frontend*"
  "*openvino_onnx_frontend*")

# Collect all files in DIR matching _OVEP_OV_PATTERNS into OUT_VAR.
function(_ovep_glob_ov _out_var _dir)
  set(_result "")
  foreach(_pat IN LISTS _OVEP_OV_PATTERNS)
    file(GLOB _tmp "${_dir}/${_pat}")
    list(APPEND _result ${_tmp})
  endforeach()
  set(${_out_var} "${_result}" PARENT_SCOPE)
endfunction()

if(DEFINED ENV{INTEL_OPENVINO_DIR})
  # Archive / installer layout
  if(WIN32)
    set(_ov_bin  "${INTEL_OPENVINO_DIR}/runtime/bin/intel64")
    set(_tbb_bin "${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/bin")

    # TBB: split into non-debug (Release/RelWithDebInfo) and debug subsets
    file(GLOB _all_tbb "${_tbb_bin}/tbb*.dll")
    set(_tbb_release "")
    set(_tbb_debug   "")
    foreach(_f IN LISTS _all_tbb)
      get_filename_component(_lname "${_f}" NAME)
      string(TOLOWER "${_lname}" _lname_lower)
      if(_lname_lower MATCHES "debug")
        list(APPEND _tbb_debug   "${_f}")
      else()
        list(APPEND _tbb_release "${_f}")
      endif()
    endforeach()

    if(EXISTS "${_ov_bin}/Release")
      _ovep_glob_ov(_ov_release "${_ov_bin}/Release")
      set(OVEP_OV_TBB_INSTALL_FILES_Release ${_ov_release} ${_tbb_release})
    else()
      message(WARNING "OpenVINO Release binaries not found at: ${_ov_bin}/Release")
    endif()

    if(EXISTS "${_ov_bin}/Debug")
      _ovep_glob_ov(_ov_debug "${_ov_bin}/Debug")
      set(OVEP_OV_TBB_INSTALL_FILES_Debug ${_ov_debug} ${_tbb_debug})
    else()
      message(WARNING "OpenVINO Debug binaries not found at: ${_ov_bin}/Debug")
    endif()

    if(EXISTS "${_ov_bin}/RelWithDebInfo")
      _ovep_glob_ov(_ov_rwdi "${_ov_bin}/RelWithDebInfo")
      set(OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo ${_ov_rwdi} ${_tbb_release})
    elseif(DEFINED OVEP_OV_TBB_INSTALL_FILES_Release)
      message(STATUS "OpenVINO RelWithDebInfo binaries not found, falling back to Release")
      set(OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo ${OVEP_OV_TBB_INSTALL_FILES_Release})
    else()
      message(WARNING
        "OpenVINO RelWithDebInfo binaries not found and Release fallback also missing at: ${_ov_bin}")
    endif()

  else()
    # Linux: single flat lib directory, no per-config split
    set(_ov_lib  "${INTEL_OPENVINO_DIR}/runtime/lib/intel64")
    set(_tbb_lib "${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/lib")

    _ovep_glob_ov(_ov_files "${_ov_lib}")
    file(GLOB _tbb_files "${_tbb_lib}/libtbb*.so*"
                         "${_tbb_lib}/libhwloc.so*")
    set(_all_linux ${_ov_files} ${_tbb_files})

    set(OVEP_OV_TBB_INSTALL_FILES_Release        ${_all_linux})
    set(OVEP_OV_TBB_INSTALL_FILES_Debug          ${_all_linux})
    set(OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo ${_all_linux})
  endif()

else()
  # Python wheel layout — all binaries flat in <site-packages>/openvino/libs/
  set(_libs "${INTEL_OPENVINO_DIR}/libs")
  if(NOT EXISTS "${_libs}")
    message(FATAL_ERROR "OpenVINO wheel libs directory not found: ${_libs}\n"
      "Expected layout: <site-packages>/openvino/libs/")
  endif()

  _ovep_glob_ov(_ov_wheel "${_libs}")
  if(NOT _ov_wheel)
    message(FATAL_ERROR "No OpenVINO binaries found in wheel libs directory: ${_libs}")
  endif()
  if(WIN32)
    file(GLOB _tbb_wheel "${_libs}/tbb*.dll")
  else()
    file(GLOB _tbb_wheel "${_libs}/libtbb*.so*"
                         "${_libs}/libhwloc.so*")
  endif()
  set(OVEP_OV_TBB_INSTALL_FILES_wheel ${_ov_wheel} ${_tbb_wheel})
endif()

# Collect unique DLL filenames for SxS dep manifest embedding (Windows only)
if(WIN32)
  set(_all_files
    ${OVEP_OV_TBB_INSTALL_FILES_Release}
    ${OVEP_OV_TBB_INSTALL_FILES_Debug}
    ${OVEP_OV_TBB_INSTALL_FILES_RelWithDebInfo}
    ${OVEP_OV_TBB_INSTALL_FILES_wheel})
  set(OVEP_OV_TBB_DLL_NAMES "")
  foreach(_f IN LISTS _all_files)
    get_filename_component(_ext  "${_f}" EXT)
    get_filename_component(_name "${_f}" NAME)
    if(_ext STREQUAL ".dll")
      list(APPEND OVEP_OV_TBB_DLL_NAMES "${_name}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES OVEP_OV_TBB_DLL_NAMES)

  if(NOT OVEP_OV_TBB_DLL_NAMES)
    message(FATAL_ERROR "No OpenVINO or TBB DLLs found under '${INTEL_OPENVINO_DIR}'.")
  endif()

  set(_sxs_manifest_dir "${CMAKE_BINARY_DIR}/sxs_manifests")
  file(MAKE_DIRECTORY "${_sxs_manifest_dir}")
  set(_ep_file_version "${OVEP_VERSION_STRING}.0")

  # Pre-generate the SxS dep manifest for every OV/TBB DLL at configure time.
  # dep.manifest.in uses ${DLL_BASE_NAME} and ${EP_FILE_VERSION} (${VAR} syntax).
  foreach(_dll_name IN LISTS OVEP_OV_TBB_DLL_NAMES)
    get_filename_component(_dll_we "${_dll_name}" NAME_WE)
    set(DLL_BASE_NAME   "${_dll_we}")
    set(EP_FILE_VERSION "${_ep_file_version}")
    configure_file(
      "${CMAKE_CURRENT_LIST_DIR}/sxs/dep.manifest.in"
      "${_sxs_manifest_dir}/${_dll_we}.dep.manifest"
    )
  endforeach()

  # Pre-generate per-config SxS assembly manifests at configure time.
  # Each config gets only its own DLLs — listing e.g. tbb_debug.dll in a Release
  # manifest would cause SxS load failure if that file is absent from the assembly directory.
  set(_asm_configs Release Debug RelWithDebInfo)
  if(DEFINED OVEP_OV_TBB_INSTALL_FILES_wheel)
    set(_asm_configs wheel)
  endif()
  foreach(_cfg IN LISTS _asm_configs)
    set(OVEP_ASSEMBLY_FILE_ENTRIES "")
    foreach(_f IN LISTS OVEP_OV_TBB_INSTALL_FILES_${_cfg})
      get_filename_component(_ext  "${_f}" EXT)
      get_filename_component(_name "${_f}" NAME)
      if(_ext STREQUAL ".dll")
        string(APPEND OVEP_ASSEMBLY_FILE_ENTRIES "  <file name=\"${_name}\" />\n")
      endif()
    endforeach()
    configure_file(
      "${CMAKE_CURRENT_LIST_DIR}/sxs/assembly.manifest.in"
      "${_sxs_manifest_dir}/openvino_runtime_${_cfg}.manifest"
      @ONLY)
    set(OVEP_ASSEMBLY_MANIFEST_${_cfg} "${_sxs_manifest_dir}/openvino_runtime_${_cfg}.manifest")
  endforeach()

  # ---------------------------------------------------------------------------
  # Build-time staging: copy OV/TBB files to <build>/sxs_staging/<cfg>/,
  # strip Authenticode signatures, and embed SxS dep manifests into DLLs.
  # Sets OVEP_OV_TBB_STAGED_FILES_* for use by cmake/openvino_install.cmake.
  #
  # _cfg may be a literal config name (Release/Debug/RelWithDebInfo) or $<CONFIG>.
  # Returns the destination path in <out_dst>.
  # ---------------------------------------------------------------------------
  set(_sxs_staging_root "${CMAKE_BINARY_DIR}/sxs_staging")

  function(_ovep_stage_file _out_dst _cfg _src)
    get_filename_component(_name "${_src}" NAME)
    get_filename_component(_ext  "${_src}" EXT)
    set(_dst "${_sxs_staging_root}/${_cfg}/${_name}")
    if(_ext STREQUAL ".dll")
      get_filename_component(_name_we "${_src}" NAME_WE)
      add_custom_command(
        OUTPUT "${_dst}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_sxs_staging_root}/${_cfg}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${_src}" "${_dst}"
        COMMAND "${CMAKE_COMMAND}"
          "-DSIGNTOOL=${SIGNTOOL}"
          "-DDLL_PATH=${_dst}"
          -P "${CMAKE_CURRENT_LIST_DIR}/sxs/remove_signature.cmake"
        COMMAND "${CMAKE_COMMAND}"
          "-DMT=${MT}"
          "-DDLL_PATH=${_dst}"
          "-DMANIFEST_PATH=${_sxs_manifest_dir}/${_name_we}.dep.manifest"
          -P "${CMAKE_CURRENT_LIST_DIR}/sxs/embed_manifest.cmake"
        DEPENDS "${_src}" "${_sxs_manifest_dir}/${_name_we}.dep.manifest"
        COMMENT "SxS ${_cfg}/${_name}: remove signature + embed manifest"
        VERBATIM
      )
    else()
      add_custom_command(
        OUTPUT "${_dst}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_sxs_staging_root}/${_cfg}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${_src}" "${_dst}"
        DEPENDS "${_src}"
        COMMENT "SxS ${_cfg}/${_name}: copy"
        VERBATIM
      )
    endif()
    set(${_out_dst} "${_dst}" PARENT_SCOPE)
  endfunction()

  if(DEFINED OVEP_OV_TBB_INSTALL_FILES_wheel)
    set(_staged_wheel "")
    foreach(_src IN LISTS OVEP_OV_TBB_INSTALL_FILES_wheel)
      _ovep_stage_file(_dst "$<CONFIG>" "${_src}")
      list(APPEND _staged_wheel "${_dst}")
    endforeach()
    set(OVEP_OV_TBB_STAGED_FILES_wheel "${_staged_wheel}")
    add_custom_target(ovep_embed_ov_tbb_manifests ALL DEPENDS ${OVEP_OV_TBB_STAGED_FILES_wheel})

  else()
    set(_all_staged "")
    foreach(_cfg IN ITEMS Release Debug RelWithDebInfo)
      foreach(_src IN LISTS OVEP_OV_TBB_INSTALL_FILES_${_cfg})
        _ovep_stage_file(_dst "${_cfg}" "${_src}")
        list(APPEND _all_staged "${_dst}")
        list(APPEND OVEP_OV_TBB_STAGED_FILES_${_cfg} "${_dst}")
      endforeach()
    endforeach()
    add_custom_target(ovep_embed_ov_tbb_manifests ALL DEPENDS ${_all_staged})
  endif()
endif()
