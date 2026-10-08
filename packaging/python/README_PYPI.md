# OpenVINO™ Execution Provider Plugin for ONNX Runtime

[![PyPI version](https://badge.fury.io/py/onnxruntime-ep-openvino.svg)](https://pypi.org/project/onnxruntime-ep-openvino/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides the [OpenVINO™ Execution Provider](https://docs.openvino.ai/) plugin for [ONNX Runtime](https://onnxruntime.ai/).

## Installation

### Prerequisites

1. **ONNX Runtime**: Install separately (this package does NOT include ONNX Runtime)
   ```bash
   pip install onnxruntime>=1.23.0
   ```

2. **Platform**: Pre-built wheels available for:
   - Windows x64
   - Linux x86_64 (manylinux_2_28 compatible — glibc 2.28+, e.g. RHEL 8, Ubuntu 20.04+)

### Install the Plugin

```bash
pip install onnxruntime-ep-openvino
```

**Note:** This package is distributed as binary wheels only. If no wheel is available for your platform/Python version, installation will fail. Source builds are not supported.

## Usage

### Basic Example

```python
import onnxruntime as ort
import onnxruntime_ep_openvino as openvino_ep

# Get the plugin library path
ep_lib_path = openvino_ep.get_library_path()

# Register the plugin with ONNX Runtime
registration_name = "openvino_ep"
ort.register_execution_provider_library(registration_name, ep_lib_path)

# Get the EP name
ep_name = openvino_ep.get_ep_name()

# Get available OpenVINO devices
all_ep_devices = ort.get_ep_devices()
openvino_devices = [d for d in all_ep_devices if d.ep_name == ep_name]

if not openvino_devices:
    raise RuntimeError("No OpenVINO devices found")

# Select OpenVINO CPU device
cpu_devices = [d for d in openvino_devices if d.ep_metadata.get("ov_device") == "CPU"]

if not cpu_devices:
    raise RuntimeError("No OpenVINO CPU device found")

# Create session options
sess_options = ort.SessionOptions()

# Add OpenVINO EP to the session
ep_options = {}
sess_options.add_provider_for_devices([cpu_devices[0]], ep_options)

# Create inference session
model_path = "path/to/model.onnx"
session = ort.InferenceSession(model_path, sess_options=sess_options)

# Run inference
# ... your inference code here ...

# Clean up
del session

# Unregister the library after all sessions are closed
ort.unregister_execution_provider_library(registration_name)
```

### Helper Functions

#### `get_library_path()`
Returns the full path to the OpenVINO EP plugin library.

```python
import onnxruntime_ep_openvino as openvino_ep

lib_path = openvino_ep.get_library_path()
print(f"Plugin library: {lib_path}")
```

#### `get_ep_name()`
Returns the name of the Execution Provider: `"OpenVINOExecutionProvider"`

```python
ep_name = openvino_ep.get_ep_name()
print(f"EP name: {ep_name}")
```

#### `get_ep_names()`
Returns a list containing the EP name (for compatibility with multi-EP plugins).

```python
ep_names = openvino_ep.get_ep_names()
# Returns: ['OpenVINOExecutionProvider']
```

## Troubleshooting

### Library Not Found
If you get a `FileNotFoundError`, ensure:
1. The package is properly installed: `pip list | grep onnxruntime-ep-openvino`
2. Reinstall if needed: `pip install --force-reinstall onnxruntime-ep-openvino`

Note: OpenVINO and TBB libraries are bundled with the package - no separate installation needed!

### No Devices Found
If `get_ep_devices()` doesn't return OpenVINO devices:
1. Check that the plugin is registered before calling `get_ep_devices()`
2. Verify your hardware is supported by OpenVINO
3. Restart your Python session after installation

### Version Compatibility
- Requires ONNX Runtime 1.23.0 or later

## License

This project is licensed under the MIT License

---

**Copyright (C) Intel Corporation**
