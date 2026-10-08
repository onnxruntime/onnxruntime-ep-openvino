# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
OpenVINO™ Execution Provider Plugin for ONNX Runtime

Provides the OpenVINO™ Execution Provider plugin for ONNX Runtime,
enabling hardware-accelerated inference on Intel® CPUs, GPUs, and NPUs.
"""

import os
import sys
from pathlib import Path
from typing import List

# EP name constant - matches kOpenVINOExecutionProvider in C++ code
_EP_NAME = "OpenVINOExecutionProvider"

# Library name constants
if sys.platform == "win32":
    _LIBRARY_NAME = "onnxruntime_providers_openvino_plugin.dll"
else:
    _LIBRARY_NAME = "libonnxruntime_providers_openvino_plugin.so"


def get_library_path() -> str:
    """
    Get the path to the OpenVINO EP plugin library.
    
    Returns the full path to the plugin shared library. This path should be
    passed to onnxruntime.register_execution_provider_library().
    
    The package includes OpenVINO and TBB runtime libraries in the same
    directory. The plugin uses RPATH (Linux) or looks in its own directory
    (Windows) to find dependencies automatically.
    
    Returns:
        str: Full path to the plugin library.
        
    Raises:
        FileNotFoundError: If the plugin library cannot be found.
    """
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent.absolute()

    # Look for the library in the package directory
    library_path = package_dir / _LIBRARY_NAME
    
    if not library_path.exists():
        raise FileNotFoundError(
            f"OpenVINO EP plugin library not found at {library_path}. "
            f"Make sure the package is properly installed."
        )
    
    return str(library_path)


def get_ep_name() -> str:
    """
    Get the name of the OpenVINO Execution Provider.
    
    Returns the EP name used when filtering OrtEpDevice instances.
    
    Returns:
        str: The EP name ("OpenVINOExecutionProvider").
    """
    return _EP_NAME


def get_ep_names() -> List[str]:
    """
    Get the list of EP names supported by this plugin.
    
    Returns a list containing the single EP name, for compatibility
    with plugins that may provide multiple execution providers.
    
    Returns:
        List[str]: List containing the OpenVINO EP name.
    """
    return [_EP_NAME]


# Export public API
__all__ = [
    "get_library_path",
    "get_ep_name",
    "get_ep_names"
]
