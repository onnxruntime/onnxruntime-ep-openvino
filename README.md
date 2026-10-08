# Intel® OpenVINO™ Execution Provider Plugin

OpenVINO™ Execution Provider plugin for ONNX Runtime - enables hardware-accelerated inference on Intel® CPUs, GPUs, and NPUs.

It implements ONNX Runtime's plugin EP API (`OrtEp` / `OrtEpFactory`) as a single shared library, and can be loaded by any ONNX Runtime 1.23.0+ application.

## Table of contents

- [Building from source](#building-from-source)
  - [Build requirements](#build-requirements)
  - [Build instructions](#build-instructions)
  - [Output](#output)
  - [Running tests with CTest](#running-tests-with-ctest)
- [Provider options](#provider-options)
- [License](#license)

## Building from source

### Build requirements
* OpenVINO 2026.4+: archive/installer package **or** `pip install openvino`
* ONNX Runtime 1.30.0+: set `ONNXRUNTIME_ROOT_DIR` to the installed archive package root
* Microsoft Visual Studio 2022 (Windows) or GCC/Clang (Linux)
* CMake 3.28+
* git-lfs
* Python 3.10+ (required when using the OpenVINO wheel or building a Python wheel output)

### Build instructions

```bash
# Option 1 — OpenVINO archive/installer
/path/to/openvino/setupvars.bat   # Windows
source /path/to/openvino/setupvars.sh  # Linux

# Option 2 — OpenVINO Python wheel (no setupvars required)
pip install openvino

# Windows
set ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
cmake --preset windows-x64
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel

# Linux
export ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
cmake --preset linux-x64-release
cmake --build --preset linux-x64-release --parallel
```

### Output

Built artifacts land in `bin/windows/<config>/bin/` (Windows) or `bin/linux/<config>/lib/` (Linux), including OpenVINO/TBB runtime libraries and the plugin DLLs/SOs.

When `OVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON`, the Python wheel is produced at:
```
bin/windows/<config>/python/dist/onnxruntime_ep_openvino-*.whl
```

### Running tests with CTest

The `ovep_tests` GoogleTest cases are registered individually with CTest. Build and
install the project first, then run CTest for the selected configuration.
CTest metadata remains in the build tree, while each test runs the installed
`ovep_tests` executable from the installed runtime directory:

```bash
# Windows
ctest --test-dir build/windows -C RelWithDebInfo --parallel 4 --output-on-failure

# Linux
ctest --test-dir build/linux-x64-release --parallel 4 --output-on-failure
```

CTest launches each discovered GoogleTest case in its own process, allowing the
cases to run in parallel. Each process writes under a unique `temp/process_<pid>`
directory.

## Provider options

`hint_enable_gqa` accepts `true`, `false`, `1`, or `0` (case-insensitive),
defaulting to `false`. Set it to `true` or `1` to opt in to
`com.microsoft::GroupQueryAttention` support, including
inside control-flow subgraphs.

GQA is currently unsupported by default. This option only enables support;
`false`, `0`, or omission cannot disable support provided by the operator list or
`ORT_OPENVINO_ACCEPT_OPS`.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (C) Intel Corporation

Third-party software distributed with this project is listed, with its required notices and license terms, in [third-party-programs.txt](third-party-programs.txt).
