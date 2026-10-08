# plugin — Agent Guide

This directory contains the core EP implementation. Also read the root `AGENTS.md` for project-wide conventions.

---

## File Responsibilities

| File(s) | Responsibility |
|---------|---------------|
| `ov_factory.*` | `OpenVINOEpPluginFactory` — implements `OrtEpFactory`; enumerates devices, creates EP instances and device allocators |
| `ov_provider.*` | `OpenVINOEpPlugin` — implements `OrtEp`; checks graph capability and compiles subgraphs to OpenVINO IR |
| `ov_compute.*` | `OvComputeInfo` — implements `OrtNodeComputeInfo`; manages the `InferRequestPool` and executes inference |
| `ov_compute_stateful.*` | `OvComputeInfoStateful` — extends `OvComputeInfo` for LLM/CausalLM models with persistent KV cache state |
| `ov_bin_manager.*` | `BinManager` — serializes and deserializes compiled OpenVINO blobs for EPContext |
| `ov_ep_context.h` | `EpContextNode` — parses EPContext ONNX nodes (attributes, embedded/external blob paths) |
| `ov_shared_context.*` | `SharedContextManager` — caches compiled models across sessions keyed by device + model hash |
| `ov_supported_ops.cc` | ONNX operator whitelist used by `GetCapability` |
| `ov_plugin_utils.*` | Configuration parsing, provider option handling, device string utilities |
| `ov_version.h` | Compile-time OpenVINO version assertion |
| `transformations/` | ONNX graph transformations applied before OpenVINO compilation |

---

## Execution Flow in Detail

### GetCapability
1. Walk the graph nodes.
2. For each node check `ov_supported_ops.cc` (plus any device-specific exclusions).
3. Group supported consecutive nodes into subgraphs; return as `OrtIndexedSubGraph` list.
4. EPContext nodes short-circuit — they are always accepted when EPContext is enabled.

### Compile
1. Receive the list of subgraphs from `GetCapability`.
2. Feed each `OrtGraph` subgraph to the OpenVINO ONNX frontend via a `GraphIterator` delegate (`ov_onnx_delegate.h`) — no intermediate `onnx::ModelProto` is constructed.
3. Apply transformations (`transformations/`).
4. Optionally dump pre/post-transformation models for debug.
5. Convert to `ov::Model` through the ONNX frontend.
6. Compile with `ov::Core::compile_model()` for the target device.
7. Wrap in `OvComputeInfo` (or `OvComputeInfoStateful`) and return `OrtNodeComputeInfo`.

### Compute
1. Pull an `ov::InferRequest` from `InferRequestPool` (blocks if all are in use).
2. Bind ORT input tensors to OV tensors via `OnnxToOvNetworkBindings`.
3. Call `infer_request.infer()`.
4. Copy or alias output tensors back to ORT output buffers.
5. Return the `InferRequest` to the pool.

---

## API Entry Point Pattern

Every function pointer on `OrtEp` / `OrtEpFactory` / `OrtNodeComputeInfo` must be set in the constructor using `OVEP_API_ENTRY`:

```cpp
// In constructor body:
OrtEp::Compile = OVEP_API_ENTRY(OpenVINOEpPlugin, Compile);
OrtEp::GetName = OVEP_API_ENTRY(OpenVINOEpPlugin, GetName);
```

`OVEP_API_ENTRY` expands to a `noexcept` static thunk that:
- Casts the opaque `this_ptr` to `OpenVINOEpPlugin*`
- Calls the member function
- Catches all exceptions and converts them to `OrtStatus*`
- Logs errors via `Ort::Logger` if the derived class exposes `GetLogger()`

Use `OVEP_API_ENTRY_EX` when you need a non-default `OrtErrorCode`.

**Never** write manual thunk functions or duplicate this catch/log/return pattern.

---

## ORT Status Rules

Functions that implement ORT interface callbacks must return `OrtStatus*`:
- Return `nullptr` to indicate success.
- Use `OVEP_RETURN_IF_ERROR` to propagate a callee's status.
- Use `OVEP_RETURN_IF` / `OVEP_MAKE_STATUS` to produce new error statuses.
- Never throw; the `ApiEntry` template is the only exception-to-status translation boundary.

Internal helpers and constructors may throw (`OVEP_THROW`, `OVEP_ENFORCE`). The `ApiEntry` wrapper catches these at the API boundary.

---

## Configuration Structures

Provider options flow through a hierarchy defined in `ov_provider.h`:

```
OpenVINOEpPluginOptions
 ├─ EpContextSessionConfig     — EPContext enable/embed/path flags
 └─ OpenVINOEpProviderOptions
     ├─ ConfigMap load_config_  — raw key/value pairs forwarded to OV Core
     ├─ bool enable_causallm_   — opt-in to OvComputeInfoStateful
     └─ ReshapeMap reshape_input_ — per-input shape overrides at runtime
```

Parse provider options in `ov_plugin_utils.cc`. Validate and document any new option key in that file.

### Environment-Level vs. Session-Level Config Keys

`GetSupportedDevices` runs once per process in `CreateEpFactories`, before any session exists, so
it can't read session-level options. Keys that affect enumeration itself (e.g.
`ep.openvinoexecutionprovider.hardware_device_type`, ORT's `allow_virtual_devices`) are env-level
config entries instead: read via `Ort::GetEnvConfigEntries()`, set through
`OrtEnvCreationOptions::config_entries` at `OrtEnv` creation, not `AddSessionConfigEntry`.

---

## Adding a New Supported Operator

1. Add the op type (and domain, if non-default) to the appropriate table in `ov_supported_ops.cc`.
2. Add a test case in `tests/src/onnxruntime-abi/` that exercises the new op.
3. If the op requires a minimum OpenVINO version, gate it with a version check in `ov_version.h`.

---

## Thread Safety Requirements

- `ov::Core` is shared across sessions via `WeakSingleton<ov::Core>`. Consult the OpenVINO documentation for the thread-safety guarantees of each `ov::Core` method before calling it from multiple threads concurrently.
- `InferRequestPool` is thread-safe by design — do not access its `ov::InferRequest` objects outside the pool protocol.
- `SharedContextManager` uses a mutex to guard its compiled-model map.
- Do not store mutable session-specific state in factory-level objects.

---

## Memory Ownership

- `OvComputeInfo` **owns** its `ov::CompiledModel` and `InferRequestPool`.
- `SharedContextManager` may vend shared ownership (`std::shared_ptr`) of a compiled model; `OvComputeInfo` may hold a shared copy.
- Input/output tensors passed by ORT are **borrowed** — never store raw pointers to them beyond the scope of a single `Compute()` call.
- Use `gsl::narrow<T>` for any integer narrowing conversions.
