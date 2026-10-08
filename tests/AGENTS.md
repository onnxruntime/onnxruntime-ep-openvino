# tests — Agent Guide

Test-specific guidance for the OpenVINO EP test suite. Project-wide rules are in the root `AGENTS.md`.

## Running Tests

Run from `bin/<config>/bin/`, where the test binary and plugin DLLs are installed:

```bash
# Full suite
./ovep_tests.exe

# Full suite while allowing unavailable platform resources
OVEP_TESTS_SKIP_UNAVAILABLE=1 ./ovep_tests.exe

# Focused debugging
./ovep_tests.exe --gtest_filter=*ort_abi_ep_context*
```

Filters are for focused debugging only. Do not use them to hide GPU, NPU, cross-device, or cross-EP cases.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `OVEP_TESTS_SKIP_UNAVAILABLE` | Exactly `1` skips tests with unavailable devices, virtual targets, compatibility capabilities, or legacy EP. Any other value keeps them failing. |
| `OVEP_TESTS_ORT_LOG_LEVEL` | Sets ORT logging verbosity (`VERBOSE`, `INFO`, `WARNING`, `ERROR`, or `FATAL`). Default: `WARNING`. Redirect `VERBOSE` output to a file. |
| `OVEP_TESTS_HARDWARE_DEVICE_TYPE_FILTER` | Sets `ep.openvinoexecutionprovider.hardware_device_type` on the process's `OrtEnv` (`tests/src/main.cpp`). Env-level EP config can only be set on the first `OrtEnv` created in a process, so this only takes effect for a fresh `ovep_tests.exe` invocation, not mid-suite. |
| `OVEP_TESTS_ALLOW_VIRTUAL_DEVICES` | If set (to any value), sets ORT's `allow_virtual_devices` on the process's `OrtEnv` (`tests/src/main.cpp`). Same first-`OrtEnv`-only caveat as above. |

The default availability behavior is intentionally strict so missing or inapplicable test resources remain visible.

## Writing Tests

### Fixtures and Device Availability

- New ABI tests should use `ortAbiTestFixture` from `include/fixtures/ortAbiTestFixture.h`.
- Tests parameterized only by one physical device should inherit `ortAbiSingleDeviceTestFixture`. Use
  `ortAbiSingleDeviceTestFixture::all_devices()` for the standard CPU/GPU/NPU matrix; keep special, tuple, and
  meta-device parameterizations explicit.
- `ortAbiTestFixture::SetUp` always requires CPU and checks devices returned by
  `additional_required_devices()`. Physical and meta-device names such as `AUTO:GPU` are resolved to their
  underlying devices.
- Cross-EP fixtures must override `requires_legacy_ep()`. The legacy provider is checked for loadability, not merely
  file existence.
- Tests outside `ortAbiTestFixture` must use `include/common/test_availability.h` and follow the same contract. An
  enumerated device that cannot perform the operation under test is unavailable.
- Report unavailable or inapplicable resources with `FAIL()` by default; use `GTEST_SKIP()` only when
  `OVEP_TESTS_SKIP_UNAVAILABLE` is exactly `1`. Do not add unconditional skips for platform requirements.
- Keep CPU/GPU/NPU, cross-device, and cross-EP cases parameterized and collected.

### Version Gating

- Use `OVEP_SKIP_IF_OV_BELOW(major, minor)` for OpenVINO version requirements.
- Use `OVEP_SKIP_IF_ORT_API_BELOW(version)` for ORT API requirements.
- Tests for version helpers and skip macros belong in `src/self_tests/`.

### Test Data and Output

- Prefer the smallest existing model that exercises the behavior under test.
- New files in `testdata/` must be tracked with git-lfs.
- `ortAbiTestFixture` clears and recreates `temp_dir_` for each test and removes it during teardown. Write generated
  output there; do not duplicate its cleanup or hard-code absolute output paths.
- Use `ASSERT_*` for prerequisites and `EXPECT_*` for independent checks. Include useful failure messages.

### Paths

For test paths passed through ORT APIs, use `path_to_utf8()` and `path_from_utf8()` from
`include/common/path_utils.h`. Do not pass `path.string()` or construct a filesystem path from a potentially
non-ASCII `std::string`.
