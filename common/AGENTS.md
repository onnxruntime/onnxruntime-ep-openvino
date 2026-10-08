# common — Agent Guide

Shared headers and source files included/compiled by `plugin/` and `tests/`. Also read the root `AGENTS.md` for project-wide conventions.

---

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `ov_utils.h` | Core macros and the `ApiEntry` template; include this in every implementation file |
| `ov_utils.cc` | Definitions for non-inline utilities declared in `ov_utils.h` (e.g. `GetEnvironmentVar`); includes platform headers (`<windows.h>` / `<cstdlib>`) here rather than in the shared header |
| `ort_api_version.h` | ORT API version constants and runtime version negotiation logic |
| `ov_plugin_ver.h` | Plugin version string constants |
| `ov_bin_format.h` | Binary layout definitions for EPContext serialized blobs |

---

## `ov_utils.h` — Key Macros

### Error handling macros

```cpp
// Propagate OrtStatus* returned by a callee
OVEP_RETURN_IF_ERROR(expr_returning_OrtStatus_ptr);

// Return a new OrtStatus* if a condition is true (requires ort_api reference)
OVEP_RETURN_IF(condition, ort_api, "message");

// Return a new OrtStatus* using Ort::GetApi() (no explicit ort_api needed)
OVEP_RETURN_STATUS_IF(condition, ORT_EP_FAIL, "message");

// Allocate a raw OrtStatus* directly
return OVEP_MAKE_STATUS("message", ORT_EP_FAIL);

// Throw std::runtime_error (OK in constructors/helpers, never in API callbacks)
OVEP_THROW("message with ", variable);
OVEP_ENFORCE(condition, "message if false");
```

`OVEP_ERROR_STR(...)` is used internally by the macros to build error strings that include the source file and line; do not call it directly in production code.

### Copy/Move deletion helpers

```cpp
OVEP_DISABLE_COPY(ClassName)          // Deletes copy ctor and copy assignment
OVEP_DISABLE_MOVE(ClassName)          // Deletes move ctor and move assignment
OVEP_DISABLE_COPY_AND_MOVE(ClassName) // Both of the above
```

Use these in classes that manage OV or ORT resources that must not be accidentally copied.

### API entry point macro

```cpp
// Bind a member function to an ORT C struct function pointer
SomeOrtStruct::FunctionPtr = OVEP_API_ENTRY(DerivedClass, MemberFunction);

// With a non-default error code
SomeOrtStruct::FunctionPtr = OVEP_API_ENTRY_EX(DerivedClass, MemberFunction, ORT_EP_FAIL);
```

`OVEP_API_ENTRY` generates a `noexcept` static thunk that:
1. Casts the opaque `this_ptr` parameter to `DerivedClass*`
2. Invokes `DerivedClass::MemberFunction` with the remaining arguments
3. Catches `Ort::Exception`, `std::exception`, and `...`, logs via `Ort::Logger`, and converts to `OrtStatus*`

The `HasLogger` concept controls whether the logger is extracted automatically from the derived class. Implement `GetLogger() -> Ort::Logger` on your class to enable automatic error logging.

### Utility functions

| Function | Purpose |
|----------|---------|
| `AlignUp(value, alignment)` | Round `value` up to the next multiple of `alignment` (must be power of two) |
| `IsPowerOfTwo(value)` | Returns true if `value` is a non-zero power of two |
| `IsWithinBounds(offset, size, total_size)` | Overflow-safe check that `[offset, offset+size)` fits within `total_size` |
| `PathFromUtf8(utf8)` | Build `fs::path` from a UTF-8 `std::string_view`; avoids CP_ACP on Windows — use instead of `fs::path(std::string)` for any user-supplied path |
| `ToUtf8String(path)` | Serialise `fs::path` to a UTF-8 `std::string`; avoids CP_ACP on Windows — use instead of `path.string()` when the result will be stored, hashed, or passed as a file path |
| `GetEnvironmentVar(name)` | Cross-platform environment variable read; uses `GetEnvironmentVariableA` on Windows and `getenv` on POSIX; defined in `ov_utils.cc`; returns empty string if unset |
| `InitCxxApi(ort_api_base)` | One-time ORT C++ API initialisation; call once from the DLL entry point |

### Named constants

- `kOvepLogTag` — log prefix `"[OpenVINO-EP] "` prepended to all error messages
- `kOpenVINOExecutionProvider` — canonical EP name string `"OpenVINOExecutionProvider"`

---

## `ort_api_version.h` — Version Negotiation

- `ort_compile_time_api_version` — `constexpr`; equals `ORT_API_VERSION` from the ORT headers the plugin was built against.
- `ort_runtime_api_version` — negotiated at startup to `min(parsed_runtime_version, ort_compile_time_api_version)`.
- `ort_min_api_version_supported` — lowest API version accepted at runtime (currently 23).

### Gating versioned ORT features — use `OrtVersionGate`

Prefer `OrtVersionGate<N>(...)` over a bare `if (ort_runtime_api_version >= N)` — it guards at both compile-time and runtime.

Pass the ORT struct pointer when the lambda accesses new struct fields; receive it as `auto*` to keep the field access template-dependent and safe against older headers:

```cpp
OrtVersionGate<27>(static_cast<OrtEp*>(this), [this](auto* ep) {
    ep->NewField = OVEP_API_ENTRY(MyClass, MyMethod);
});
```

Omit the pointer when no new struct fields are accessed:

```cpp
OrtVersionGate<24>([&] { DoSomethingAvailableInV24(); });
```

---

## `ov_bin_format.h` — Blob Layout

Defines the binary layout written by `BinManager` into EPContext files. Treat these layout structs as a **versioned format**: changing field order or removing fields is a breaking change for existing cached models. When extending the format, increment the format version constant and maintain backward-reading compatibility in `ov_bin_manager.cc`.

---

## Adding New Shared Utilities

Before adding to `common/`, ask:
1. Is the utility genuinely needed by more than one top-level directory?
2. Does it introduce a dependency that not all consumers want?

If yes to (1) and no to (2), add it here. Otherwise keep it in the directory that owns it. Headers in `common/` must not include private plugin implementation headers.
