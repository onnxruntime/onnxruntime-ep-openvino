# OpenVINO™ Execution Provider Plugin — Agent Guide

## What This Project Is

An **ONNX Runtime Execution Provider (EP) plugin** that accelerates model inference on Intel® CPUs, GPUs, and NPUs using OpenVINO™. It implements ONNX Runtime's plugin API (`OrtEp` / `OrtEpFactory` interfaces) as a single DLL on both Windows and Linux.

All production code lives in the `onnxruntime::openvino_ep_plugin` namespace.

---

## Directory Map

```
plugin/           # Core EP implementation (factory, provider, compute, caching)
common/           # Shared headers: macros, ORT API versioning, binary format
tests/            # Google Test–based E2E and ABI compatibility tests (C++23)
cmake/            # CMake helper scripts
tools/            # Standalone utility programs (e.g., blob extractor)
packaging/python/ # Python wheel packaging
```

Each subdirectory with significant complexity has its own `AGENTS.md` with deeper guidance.

---

## Licensing

This project is MIT licensed — see `LICENSE` at the repo root. Every new file must open with the following header (comment syntax per file type):

```
Copyright (C) <year(s)> Intel Corporation

SPDX-License-Identifier: MIT
```

Vendored/adapted code keeps its upstream license header layered above the Intel line and must be listed in `third-party-programs.txt`.

---

## Build Prerequisites and Setup

| Requirement | Detail |
|-------------|--------|
| Compiler | MSVC (VS 2022) on Windows; GCC/Clang on Linux |
| CMake | 3.28+ |
| OpenVINO | 2026.3+ — run `setupvars.bat` / `setupvars.sh` before configuring |
| ONNX Runtime | 1.30.0+ — set `ONNXRUNTIME_ROOT_DIR` to the installed package root |
| git-lfs | Required for test model data in `tests/testdata/` |

```bash
# Windows
path/to/openvino/setupvars.bat
set ONNXRUNTIME_ROOT_DIR=path/to/onnxruntime

cmake --preset windows-x64
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel

# Linux
source path/to/openvino/setupvars.sh
export ONNXRUNTIME_ROOT_DIR=path/to/onnxruntime

cmake --preset linux-x64-release
cmake --build --preset linux-x64-release --parallel
```

Built artifacts land in `bin/<config>/bin/`. The test binary `ovep_tests.exe` is placed there alongside the plugin DLLs.

Optional CMake flags:
- `-DOVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON` — build the Python wheel
- `-DOVEP_PLUGIN_USE_MSVC_STATIC_RUNTIME=OFF` — link against dynamic CRT

---

## Architecture: Factory → Provider → Compute

```
OrtEpFactory  (OpenVINOEpPluginFactory)
  └─ CreateEp()  →  OrtEp  (OpenVINOEpPlugin)
                      ├─ GetCapability()  — returns supported subgraphs
                      └─ Compile()        →  OrtNodeComputeInfo  (OvComputeInfo)
                                                └─ Compute()   — runs inference
```

- The **factory** enumerates available hardware and creates per-session EP instances.
- The **provider** converts ONNX subgraphs to OpenVINO IR and produces compute nodes.
- The **compute node** holds a compiled `ov::CompiledModel` and an `InferRequestPool` for thread-safe execution.
- **Stateful models** (e.g., LLMs with KV cache) use `OvComputeInfoStateful` instead of `OvComputeInfo`.
- **Shared contexts** (`SharedContextManager`) allow compiled models to be reused across sessions.

---

## Core Coding Conventions

### C++ Standard
- Implementation (`plugin/`, `common/`): **C++23**
- Tests (`tests/`): **C++23**

### Style
- Formatting: Google C++ style enforced by `.clang-format`, 120-character column limit.
- Include guards: `#pragma once` (never `#ifndef` guards).
- Every new file must open with the copyright header — see [Licensing](#licensing).
- Naming: `PascalCase` for types, `snake_case` for variables and functions, `UPPER_SNAKE_CASE` for macros. Project macros use the `OVEP_` prefix.
- Prefer `std::unique_ptr` / `std::shared_ptr` over raw owning pointers.
- Always use braces for nested `if`/`for`/`while` bodies.

### Comments

Comments explain **why**, not **what**. If a comment is needed to say what the code does, rename or restructure instead.

- **Brief and to the point** — one or two lines. If it takes a paragraph to explain, the code needs simplifying or the rationale belongs in the commit message or a design doc.
- **Function-level** — a short block only when the purpose is not obvious from the name and signature.
- **Inline** — only for non-obvious branch conditions, loop setup, or ordering constraints.
- **Never** for standard patterns: range-`for` loops, null checks, RAII, `OVEP_RETURN_IF_ERROR` propagation.
- **Always** for workarounds, ORT/OV API quirks, and deliberate tradeoffs — a condition that looks backwards needs a reason beside it.
- Never narrate the diff (`// added null check`) or leave commented-out code; that is what git is for.

```cpp
// Good — intent a reader cannot derive from the code
// Compile both models with distinct bin paths first to establish a known cache state,
// then load concurrently to verify each session gets its own blob, not a stale shared one.

// Bad — restates the code
// Loop over the inputs and set each tensor
for (auto& input : inputs) { ... }
```

### Namespaces
All non-test code belongs to `onnxruntime::openvino_ep_plugin`. Avoid `using namespace` in headers.

---

## Error Handling — The Most Important Convention

You never write raw ORT C callbacks. Write a normal member function (usually returning `OrtStatus*`) and bind it with `OVEP_API_ENTRY` — the `ApiEntry` thunk (`common/ov_utils.h`) is the single `noexcept` boundary that casts `this`, catches `Ort::Exception` / `std::exception` / `...`, logs via the class `Ort::Logger`, and converts to the return type. An `Ort::Exception` keeps its own code; anything else uses the bound code (`ORT_RUNTIME_EXCEPTION`, or the one passed to `OVEP_API_ENTRY_EX`).

```cpp
some_struct.Callback        = OVEP_API_ENTRY(MyClass, MyMethod);
some_struct.Callback        = OVEP_API_ENTRY_EX(MyClass, MyMethod, ORT_EP_FAIL);  // non-default code
some_struct.ReleaseCallback = OVEP_API_ENTRY_TEARDOWN(MyClass, Release);          // teardown: no logging
```

Because `ApiEntry` catches everything, **throwing from a bound member is safe** — it becomes a status at the boundary, never `std::terminate`. So returning a status vs. throwing is a clarity choice: return for expected/recoverable failures, throw (`OVEP_THROW` / `OVEP_ENFORCE`) for invariant violations and in constructors/deep helpers.

```cpp
OVEP_RETURN_IF_ERROR(callee_returning_ortstatus());          // propagate a status (the workhorse)
OVEP_RETURN_IF(cond, ort_api, "message");                    // new status; needs an OrtApi in scope
OVEP_RETURN_STATUS_IF(cond, ORT_EP_FAIL, "message");         // new status via Ort::GetApi(), explicit code
return OVEP_MAKE_STATUS("message", ORT_EP_FAIL);             // allocate and return a status directly
OVEP_THROW("message with ", variable);                       // caught + converted by ApiEntry
OVEP_ENFORCE(cond, "message if false");
```

Rules:
- **Never discard an `OrtStatus*`** — a non-null status is an error that must propagate; wrap status-returning calls in `OVEP_RETURN_IF_ERROR`.
- **Do not hand-write callbacks or their try/catch** — always bind through `OVEP_API_ENTRY*`.
- **Do not mark bound members `noexcept`** — `ApiEntry` is the boundary; a `noexcept` member turns a throw into `std::terminate` (see CWE-248).
- All macros route through `OVEP_ERROR_STR` (prepends `[OpenVINO-EP]` + file/line) — don't build error strings by hand.

---

## OpenVINO API

Always use the **OpenVINO 2.0+ API** (`ov::` namespace). The legacy `InferenceEngine::` namespace must not appear in new code.

Key types:
- `ov::Core` — device management and model compilation. Shared via `WeakSingleton<ov::Core>`.
- `ov::Model` — ONNX graph converted to OpenVINO IR.
- `ov::CompiledModel` — hardware-compiled executable for a specific device.
- `ov::InferRequest` — single inference instance; pooled in `InferRequestPool`.
- `ov::Tensor` — input/output data buffer.

A static compile-time check in `plugin/ov_version.h` enforces the minimum OpenVINO version.

---

## EPContext (Compiled Model Caching)

EPContext is an ONNX extension (`com.microsoft` domain) for serializing compiled models. When enabled, the EP embeds or references OpenVINO blobs so subsequent sessions skip recompilation.

Configuration keys (passed as provider options):
| Key | Purpose |
|-----|---------|
| `ep.openvino.ep_context_enable` | Enable EPContext support |
| `ep.openvino.ep_context_embed_mode` | Embed blob in the ONNX model vs. external file |
| `ep.openvino.ep_context_file_path` | Path for the external blob file |

Implementation is in `plugin/ov_ep_context.h` and `plugin/ov_bin_manager.*`.

---

## Testing

Run from the `bin/<config>/bin/` directory after building:

```bash
# Run all tests
./ovep_tests.exe

# Filter to ABI tests on CPU
./ovep_tests.exe --gtest_filter=*abi*CPU*

# Enable verbose ORT logging (output is very large — pipe to a file, do not read directly into context)
OVEP_TESTS_ORT_LOG_LEVEL=VERBOSE ./ovep_tests.exe > test.log 2>&1
```

See `tests/AGENTS.md` for test authoring guidelines.

---

## Coverity Rules — Apply During Code Review

### CWE-248: Uncaught Exception
A `noexcept` function must not allow exceptions to escape — any uncaught throw calls `std::terminate`. Rules:
- All code that can throw inside a `noexcept` function must be inside a `try`/`catch(...)` block.
- A `noexcept` wrapper that calls a non-`noexcept` member function via a pointer/template will not catch exceptions if the callee is itself also marked `noexcept` — the `std::terminate` fires inside the callee before the outer catch runs. Do not mark a function `noexcept` if its body calls anything that can throw.

### Rule of Five (CWE-398)
If a class explicitly defines **any one** of {destructor, copy constructor, copy assignment operator, move constructor, move assignment operator}, it must explicitly define **all five** (as `= default`, `= delete`, or a real implementation). Partial compliance silently generates incorrect implicit members.
- Polymorphic classes must declare a virtual destructor (`virtual ~Foo() = default` in a base, `~Foo() override = default` in a derived class).
- Classes that own resources (raw pointers, OS handles, GPU tensors) and must not be copied should explicitly `= delete` copy constructor and copy assignment, and decide intentionally on move.

### COPY_INSTEAD_OF_MOVE
Prefer `std::move` when transferring ownership of heap-allocated types — `std::vector`, `std::map`, `std::filesystem::path`, `std::shared_ptr`, `std::string` (beyond SSO size ~15 chars) — where the source will not be used again. Specific cases:
- A by-value parameter stored into a member should be `std::move`d.
- A local variable that is the last operand before going out of scope should be `std::move`d into the destination.
- Avoid `const` on a local variable that will be returned by value — it suppresses implicit move and NRVO.
- `std::shared_ptr` copy increments and then decrements an atomic ref-count; `std::move` avoids both operations.
- Short strings within SSO limits benefit negligibly — do not add `std::move` for them.

### Null / Pointer Safety (FORWARD_NULL, NULL_RETURNS, USE_AFTER_FREE)
- Always null-check pointers and handles returned by ORT or OpenVINO APIs before dereferencing. Do not assume a non-null return even when the API documentation implies it.
- Do not access a raw pointer after the object it points to has been destroyed, released, or moved from. Prefer `std::unique_ptr` / `std::shared_ptr` over raw owning pointers to make lifetime explicit.

### Resource Management (RESOURCE_LEAK, CHECKED_RETURN)
- Every allocated resource must be released on **all** code paths, including error paths. `OrtStatus*` values returned by ORT allocators, `Ort::Value`, and similar owning handles must be wrapped in RAII or explicitly released.
- Never discard the return value of a function that signals failure. For ORT callbacks use `OVEP_RETURN_IF_ERROR` — letting an `OrtStatus*` go unexamined is a resource leak and a silent error swallow.

### Integer Safety (INTEGER_OVERFLOW, BAD_SHIFT)
- Unsigned arithmetic (`size_t`, `uint32_t`) silently wraps on underflow/overflow — validate inputs before arithmetic involving sizes or offsets. Use the `IsWithinBounds(offset, size, total)` helper in `ov_utils.h` for offset range checks.
- Use `gsl::narrow<T>` for any narrowing integer conversion; it throws on truncation rather than silently corrupting the value.
- Shift amounts must be non-negative and strictly less than the bit width of the shifted type.

### Concurrency (DEADLOCK, ORDER_REVERSAL, MISSING_LOCK)
- When multiple mutexes must be held simultaneously, always acquire them in the same global order across all call paths. Document the intended order at the point of declaration.
- Never read or write shared state without holding the appropriate mutex. Prefer `std::lock_guard` / `std::scoped_lock` over manual lock/unlock.

### Control Flow (MISSING_BREAK, DEADCODE)
- Every `switch` case that intentionally falls through to the next must be annotated with `[[fallthrough]]`; omitting it is both a Coverity defect and a readability issue.
- Remove or fix branches whose conditions are logically impossible given the guards above them; dead code indicates a logic error in the surrounding conditions.

---

## Unicode Path Handling

On Windows, `std::filesystem::path(std::string)` and `fs::path::string()` use the **ANSI code page (CP_ACP)**, not UTF-8. Any path containing non-ASCII characters (user home directories, localised model names, EPContext output paths) will be silently corrupted unless the correct helpers are used.

### Helpers (defined in `common/ov_utils.h`)

```cpp
// UTF-8 std::string → fs::path  (correct on all platforms)
fs::path PathFromUtf8(std::string_view utf8);

// fs::path → UTF-8 std::string  (correct on all platforms)
std::string ToUtf8String(const fs::path& path);
```

### Rules

| Situation | Correct | Wrong |
|-----------|---------|-------|
| Build `fs::path` from an ORT config value or ONNX attribute | `PathFromUtf8(str)` | `fs::path(str)` |
| Serialise `fs::path` to string (ONNX attribute, log, hash) | `ToUtf8String(path)` | `path.string()` |
| Append an ASCII-only suffix to an existing `fs::path` | `path += "_ctx.onnx"` | `path_from_utf8(path_to_utf8(path) + "_ctx.onnx")` |
| Pass a path to an OV API with an `fs::path` overload | pass `fs::path` directly | convert to string first |
| Pass a path to an interface fixed as `std::string` (e.g. OV `TensorMetaInfo::m_external_location`) | `ToUtf8String(fs::path(native_str))` | `native_str.string()` |

### Code review checklist

Flag these patterns as Unicode bugs:
- `fs::path(some_string)` where `some_string` originates from an ORT config entry, ONNX attribute, or user-supplied value — use `PathFromUtf8` instead.
- `path.string()` stored, hashed, or passed as a file path — use `ToUtf8String` instead.
- Round-tripping through string to append a suffix: `path_from_utf8(path_to_utf8(p) + "_x")` — keep the Unicode component as `fs::path` and use `operator+=` for the ASCII suffix.

---

## Common Pitfalls

- **Do not mark bound members `noexcept`** — `ApiEntry` is the catch boundary; throwing from a bound member is safe, but `noexcept` turns it into `std::terminate`.
- **Do not use** `InferenceEngine::` — it is the removed legacy OpenVINO API.
- **Do not assume** `ov::Core` or `ov::CompiledModel` are free-threaded; synchronize access or use pools.
- **Do not hardcode** file paths or device names; derive them from configuration.
- **Do not change** ORT interface struct signatures without gating on `ort_runtime_api_version`.
- **Do not conflate** the compile-time build requirement (checked via `static_assert` in `common/ort_api_version.h`) with the runtime minimum (`ort_min_api_version_supported`). The build requires a minimum ORT header version, but the plugin still supports loading against older ORT runtimes down to the `ort_min_api_version_supported` floor. Do not raise `ort_min_api_version_supported` just because the build minimum changed.
- **Do not add** comments that restate the code, narrate the diff, or preserve dead code — see the Comments conventions above.
- **Do run** `clang-format` before committing; CI will enforce formatting.
- **Do not call** `path.string()` or `fs::path(std::string)` for paths that may contain non-ASCII characters — use `ToUtf8String` / `PathFromUtf8` from `common/ov_utils.h`.

---

## Debugging Quick Reference

| Problem | First step |
|---------|------------|
| DLL load failure | Verify `setupvars.bat` ran; check all OV DLLs are present in `bin/` |
| Inference failure | Set `OVEP_TESTS_ORT_LOG_LEVEL=VERBOSE`; check `ov_supported_ops.cc` for the failing op |
| Build configuration error | Confirm `ONNXRUNTIME_ROOT_DIR` is set and points to a valid ORT package |
| Stale build artifacts | Delete `build/` and re-run `cmake --preset` |
