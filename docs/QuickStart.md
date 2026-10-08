<!--
 Copyright (C) 2026 Intel Corporation
 SPDX-License-Identifier: MIT
-->

# OpenVINO™ Plugin EP Quick Start

The Intel® OpenVINO™ Execution Provider (EP) plugin is a single shared library that implements
ONNX Runtime's plugin EP API (`OrtEp` / `OrtEpFactory`). Any ONNX Runtime **1.23.0+** application can
load it at runtime — no custom ONNX Runtime build required — to run inference on Intel CPUs, GPUs
and NPUs.

This is a quick start. For deeper detail see the
[OpenVINO documentation](https://docs.openvino.ai/) and the
[ONNX Runtime EP docs](https://onnxruntime.ai/docs/execution-providers/).

- [Install](#install)
- [Build instructions](#build-instructions)
- [Running](#running)
  - [C++ API](#c-api)
  - [Python API](#python-api)
- [Provider options](#provider-options)
- [Device support](#device-support)
- [Device memory allocator](#device-memory-allocator)
- [Operator support](#operator-support)
- [EP context model and EP context bin](#ep-context-model-and-ep-context-bin)
- [Performance test](#performance-test)
- [Profiling](#profiling)
- [Troubleshooting](#troubleshooting)

---

## Install

The fastest path — prebuilt wheels bundle the plugin plus the OpenVINO and TBB runtime libraries:

```bash
pip install onnxruntime
pip install onnxruntime-ep-openvino
```

Prebuilt wheels are available for Windows x64 and Linux x86_64 (manylinux_2_28: glibc 2.28+).
Skip to [Running](#running) if you install this way; build from source only if you need a
development build.

NuGet packages are available in [Intel.ML.OnnxRuntime.EP.OpenVINO](https://www.nuget.org/packages/Intel.ML.OnnxRuntime.EP.OpenVINO).

---

## Build instructions

### Build requirements

| Requirement | Detail |
|-------------|--------|
| OpenVINO | 2026.4+ — archive/installer package **or** `pip install openvino` |
| ONNX Runtime | 1.30.0+ — set `ONNXRUNTIME_ROOT_DIR` to the installed archive package root |
| Compiler | Microsoft Visual Studio 2022 (Windows) or GCC/Clang (Linux) |
| CMake | 3.28+ |
| git-lfs | Required for the test models in `tests/testdata/` |
| Python | 3.10+ (needed for the OpenVINO wheel and for building a Python wheel) |

### Build

```bash
# Option 1 — OpenVINO archive/installer
/path/to/openvino/setupvars.bat         # Windows
source /path/to/openvino/setupvars.sh   # Linux

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

Useful optional flags: `-DOVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON` (build the Python wheel),
`-DOVEP_PLUGIN_USE_MSVC_STATIC_RUNTIME=OFF` (link the dynamic CRT).

### Output

Artifacts land in `bin/windows/<config>/bin/` (Windows) or `bin/linux/<config>/lib/` (Linux),
together with the OpenVINO/TBB runtime libraries:

| Platform | Plugin library |
|----------|----------------|
| Windows | `onnxruntime_providers_openvino_plugin.dll` |
| Linux | `libonnxruntime_providers_openvino_plugin.so` |

With `OVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON` the wheel is written to
`bin/<platform>/<config>/python/dist/onnxruntime_ep_openvino-*.whl`.

### Run the tests (optional)

```bash
# Windows
ctest --test-dir build/windows -C RelWithDebInfo --parallel 4 --output-on-failure

# Linux
ctest --test-dir build/linux-x64-release --parallel 4 --output-on-failure
```

---

## Running

The plugin registers the EP name **`OpenVINOExecutionProvider`** (and
`OpenVINOExecutionProvider.AUTO` for the AUTO meta-device — see [Device support](#device-support)).
Usage is always the same three steps: register the library, pick an `OrtEpDevice`, add it to the
session options.

### C++ API

```cpp
#include "onnxruntime_cxx_api.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ovep-quickstart");

// 1. Register the plugin library with the environment.
env.RegisterExecutionProviderLibrary(
    "OpenVINOExecutionProvider",
    ORT_TSTR("onnxruntime_providers_openvino_plugin.dll"));  // .so on Linux

// 2. Pick an OpenVINO device. Each OrtEpDevice carries an "ov_device" metadata
//    entry with the OpenVINO device name: "CPU", "GPU", "GPU.1", "NPU", ...
std::vector<Ort::ConstEpDevice> selected;
for (const auto& device : env.GetEpDevices()) {
  if (std::string_view(device.EpName()) == "OpenVINOExecutionProvider") {
    const char* ov_device = device.EpMetadata().GetValue("ov_device");
    if (ov_device && std::string_view(ov_device) == "GPU") {
      selected.push_back(device);
      break;
    }
  }
}

// 3. Add the device to the session options, with optional provider options.
Ort::SessionOptions session_options;
std::unordered_map<std::string, std::string> ep_options{
    {"load_config", R"({"GPU": {"INFERENCE_PRECISION_HINT": "f16"}})"}};
session_options.AppendExecutionProvider_V2(env, selected, ep_options);

Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

Unregister the library with `env.UnregisterExecutionProviderLibrary("OpenVINOExecutionProvider")`
after all sessions using it are destroyed.

### Python API

```python
import onnxruntime as ort
import onnxruntime_ep_openvino as openvino_ep

# 1. Register the plugin library.
ort.register_execution_provider_library("openvino_ep", openvino_ep.get_library_path())

# 2. Pick an OpenVINO device ("CPU", "GPU", "NPU", ...).
ep_name = openvino_ep.get_ep_name()          # "OpenVINOExecutionProvider"
devices = [d for d in ort.get_ep_devices()
           if d.ep_name == ep_name and d.ep_metadata.get("ov_device") == "NPU"]
if not devices:
    raise RuntimeError("No OpenVINO NPU device found")

# 3. Add the device to the session options, with optional provider options.
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(devices, {"dump_subgraphs": "./ovep_subgraphs"})

session = ort.InferenceSession("model.onnx", sess_options=sess_options)
outputs = session.run(None, {"input": input_array})

del session
```

If you built the plugin yourself, replace `openvino_ep.get_library_path()` with the path to
`onnxruntime_providers_openvino_plugin.dll` / `libonnxruntime_providers_openvino_plugin.so` in your
build output.

---

## Provider options

Provider options are the key/value pairs passed to `AppendExecutionProvider_V2` (C++) or
`add_provider_for_devices` (Python). All values are strings.

| Option | Value | Default | Description |
|--------|-------|---------|-------------|
| `load_config` | JSON string, or path to a `.json` file | `{}` |  Enables fine-grained control over OpenVINO inference behavior by loading properties from a JSON String. This is the preferred method for configuring advanced OpenVINO features, offering: Direct access to OpenVINO runtime properties, Device-specific configuration. |
| `enable_causallm` | `true` / `false` / `1` / `0` | `false` | Use the stateful (KV-cache) execution path for causal LM models — required for ORT GenAI LLM pipelines. |
| `reshape_input` | `name[dim,dim,...]`, comma-separated | unset | Reshape graph inputs at load time. Ranges are allowed: `input_1[1..5,2,3..4],data[1,2,3]`. Useful to provide ranged dynamic dimensions on NPU. |
| `dump_subgraphs` | directory path | unset | Dump each subgraph handed to OpenVINO as an IR `.xml`/`.bin` pair (post-transformation) into this directory, which is created if missing. Debug aid; a non-empty value enables it. |

See [Query Device Properties — Configuration](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html)
for the full property list.

> **Migrating from the legacy (non-plugin) OpenVINO EP:** `device_type`, `device_id` and
> `device_luid` are ignored — select the device by passing the matching `OrtEpDevice` instead.
> `cache_dir`, `precision`, `num_of_threads`, `num_streams`, `model_priority`,
> `enable_opencl_throttling` and `enable_qdq_optimizer` are ignored — set the equivalent OpenVINO
> property through `load_config`. The plugin logs a warning naming the replacement for each.

---

## Device support

The plugin enumerates one `OrtEpDevice` per detected Intel device; the `ov_device` EP metadata
entry holds the OpenVINO device name you match on.

| `ov_device` | Hardware | Notes | OpenVINO docs |
|-------------|----------|-------|---------------|
| `CPU` | Intel® Core/Xeon CPUs | Always available | [CPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) |
| `GPU`, `GPU.1`, … | Intel integrated and discrete GPUs (Arc™, Iris® Xe) | Requires the Intel graphics driver; `GPU` is the first device | [GPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) |
| `NPU` | Intel® AI Boost NPU (Core™ Ultra) | Requires the NPU driver; static shapes preferred — see `reshape_input` | [NPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) |

**AUTO** — OpenVINO's automatic device selection is exposed as a separate EP name,
`OpenVINOExecutionProvider.AUTO`. Select one or more devices reported by that EP name and pass
them together; OpenVINO picks and switches between them (for example `AUTO:GPU,CPU`):

```python
auto_devices = [d for d in ort.get_ep_devices()
                if d.ep_name == "OpenVINOExecutionProvider.AUTO"
                and d.ep_metadata.get("ov_device") in ("GPU", "CPU")]
sess_options.add_provider_for_devices(auto_devices, {})
```

See [Automatic Device Selection](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html)
and [Inference Devices and Modes](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html).

> AUTO is not supported together with EP context models — compile for a specific device instead.

---

## Device memory allocator

Every OpenVINO EP device exposes its own host-accessible allocator (`OpenVINO_shared_CPU`,
`OpenVINO_shared_GPU`, `OpenVINO_shared_NPU`). Allocating inputs from it avoids a staging copy per
inference: worth doing for large, high-dimensional inputs (image/video tensors, long sequences),
negligible for small ones.

Take the memory info off the `OrtEpDevice` you already selected and get the allocator for it — the
same code works for CPU, GPU and NPU:

`OrtDeviceMemoryType` ranks `DEFAULT` > `HOST_ACCESSIBLE` > CPU on performance. `DEFAULT` is not host/CPU accessible — use `HOST_ACCESSIBLE` when the application touches the data.

```cpp
// ep_device is the OrtEpDevice picked in Running — any ov_device.
Ort::ConstMemoryInfo mem_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
Ort::UnownedAllocator alloc = env.GetSharedAllocator(mem_info);

std::vector<int64_t> shape{1, 3, 1024, 1024};
Ort::Value input = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
std::memcpy(input.GetTensorMutableData<float>(), src, byte_size);  // fill in place

Ort::IoBinding binding(session);
binding.BindInput("input", input);
binding.BindOutput("output", mem_info);
session.Run(Ort::RunOptions{}, binding);
```


---

## Operator support

The plugin claims the ONNX ops it can map to OpenVINO (plus selected `com.microsoft` and
`org.openvinotoolkit` ops); anything else stays on the CPU EP, so unsupported ops cost performance,
not correctness.

Per-operator, per-opset results: [ONNX Backend Scoreboard — onnxruntime-openvino](https://onnx.ai/backend-scoreboard/onnxruntime-openvino_details_stable.html).

---

## EP context model and EP context bin

An **EP context model** is an ONNX model whose OpenVINO subgraphs have been replaced by
precompiled OpenVINO blobs (`EPContext` nodes, `com.microsoft` domain). Compile once, then every
later session skips graph conversion and device compilation — the big win is session creation time,
especially on NPU. Please look at [ORT EP Context Session Options Documentation](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#onnxruntime-session-options-related-to-ep-context-cache-generation-and-inference)

### Generate (AOT)

Python, with the compile API:

```python
import onnxruntime as ort
import onnxruntime_ep_openvino as openvino_ep

ort.register_execution_provider_library("openvino_ep", openvino_ep.get_library_path())
devices = [d for d in ort.get_ep_devices()
           if d.ep_name == openvino_ep.get_ep_name() and d.ep_metadata.get("ov_device") == "NPU"]

sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(devices, {})

compiler = ort.ModelCompiler(sess_options, "model.onnx",
                             embed_compiled_data_into_model=False)  # external .bin
compiler.compile_to_file("model_ctx.onnx")
```

C++:

```cpp
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetInputModelPath(ORT_TSTR("model.onnx"));
compile_options.SetOutputModelPath(ORT_TSTR("model_ctx.onnx"));
compile_options.SetEpContextEmbedMode(false);   // external .bin
Ort::CompileModel(env, compile_options);
```

Equivalently, set `ep.context_enable=1` on the session options and just create a session — the
context model is written as a side effect.

### The EP context bin

With `ep.context_embed_mode=0` the compiled OpenVINO blobs are written to a single sidecar file
next to the generated model:

```
model_ctx.onnx                              # graph with EPContext nodes
model_ctx_OpenVINOExecutionProvider.bin      # the compiled OpenVINO model
```

The `.bin` name is derived from the context model name; each `EPContext` node references it by
**relative** file name, so keep the `.onnx` and the `.bin` in the same directory when you ship them.
The file is versioned — a major-version change in the on-disk format makes older `.bin` files unreadable, and loading the context model fails until you regenerate it from the original model. Blobs are device- and driver-specific: regenerate after changing hardware, driver, or OpenVINO version.

### Run (JIT)

Nothing special — load the context model exactly like the original:

```python
session = ort.InferenceSession("model_ctx.onnx", sess_options=sess_options)
```

```cpp
Ort::Session session(env, ORT_TSTR("model_ctx.onnx"), session_options);
```

---

## Performance test

`onnxruntime_perf_test` (shipped with ONNX Runtime) drives the plugin through the plugin-EP flags.

List what the plugin sees:

```
onnxruntime_perf_test.exe --plugin_ep_libs "OpenVINOExecutionProvider|C:\path\to\onnxruntime_providers_openvino_plugin.dll" --list_ep_devices
```

Benchmark on a chosen device (`-I` skip input generation from ONNX test data, `-t 5` run for 5
seconds, `-r N` run N iterations):

```
onnxruntime_perf_test.exe ^
  --plugin_ep_libs "OpenVINOExecutionProvider|C:\path\to\onnxruntime_providers_openvino_plugin.dll" ^
  --plugin_eps OpenVINOExecutionProvider ^
  --filter_ep_devices "ov_device|GPU" ^
  -I -t 5 model.onnx
```

With provider options (`key|value` pairs, space separated; point `load_config` at a `.json` file to
avoid quoting a JSON blob on the command line):

```
onnxruntime_perf_test.exe ... --plugin_ep_options "load_config|C:\path\to\ov_config.json"
```

Generate an EP context model through perf test, then benchmark it:

```
onnxruntime_perf_test.exe ^
  --plugin_ep_libs "OpenVINOExecutionProvider|C:\path\to\onnxruntime_providers_openvino_plugin.dll" ^
  --plugin_eps OpenVINOExecutionProvider --filter_ep_devices "ov_device|NPU" ^
  -I -r 1 --compile_ep_context --compile_model_path model_ctx.onnx model.onnx

onnxruntime_perf_test.exe ^
  --plugin_ep_libs "OpenVINOExecutionProvider|C:\path\to\onnxruntime_providers_openvino_plugin.dll" ^
  --plugin_eps OpenVINOExecutionProvider --filter_ep_devices "ov_device|NPU" ^
  -I -t 5 model_ctx.onnx
```

Use `--select_ep_devices "<index>;<index>"` with the indices printed by `--list_ep_devices` when
`--filter_ep_devices` is ambiguous (for example two GPUs).

---

## Profiling

The plugin implements ONNX Runtime's EP profiler, so OpenVINO per-node counters land in the normal
ORT profile (a Chrome trace JSON). Profiling must be enabled during compilation. 

```python
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True          # must be set BEFORE the session is created
sess_options.add_provider_for_devices(devices, {})

session = ort.InferenceSession("model.onnx", sess_options=sess_options)
session.run(None, feeds)

profile_file = session.end_profiling()        # e.g. onnxruntime_profile__2026-09-25_....json
```

```cpp
Ort::SessionOptions session_options;
session_options.EnableProfiling(ORT_TSTR("ovep_profile"));
// ... run ...
Ort::AllocatorWithDefaultOptions allocator;
auto profile_file = session.EndProfilingAllocated(allocator);  // AllocatedStringPtr
```

Load the JSON in `chrome://tracing`, `edge://tracing` or [Perfetto](https://ui.perfetto.dev/).
OpenVINO kernels appear as `KERNEL` events named `OV::<node_name>`, nested under the ORT node
event they belong to, with these args:

| Arg | Meaning |
|-----|---------|
| `ep` | Always `OpenVINOExecutionProvider` |
| `ov_node_type` | OpenVINO node type |
| `ov_exec_type` | Executed kernel implementation (e.g. `jit_avx512_...`) |
| `ov_status` | `EXECUTED`, `NOT_RUN`, `OPTIMIZED_OUT` |
| `ort_correlation_id` | Links the OV event to its ORT node event |
| `parent_ort_event_name` | Name of the enclosing ORT event |

Enabling profiling turns on OpenVINO's `PERF_COUNT` for the compiled model, which adds overhead —
measure throughput with profiling off. 

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Library fails to load | Windows: the OpenVINO/TBB DLLs must sit next to the plugin DLL (they do in the wheel and in `bin/`). |
| `get_ep_devices()` returns no OpenVINO device | Register the library first; verify the GPU/NPU driver is installed and that OpenVINO itself sees the device. |
| Model runs but falls back to CPU EP | Unsupported ops are partitioned away; dump them with `dump_subgraphs` and check the ORT log for partition messages. |
| EP context model fails to load ("Could not deserialize binary data" / bin not found) | The `.bin` is missing, moved, or was produced by an incompatible format/driver/OpenVINO version — regenerate the context model. |
| NPU compilation fails on dynamic shapes | Bound the dynamic dimensions with `reshape_input`. |

---

Copyright (C) Intel Corporation — licensed under the [MIT License](../LICENSE).
