// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

// Windows-only: verifies that the OVEP plugin loads its own openvino.dll via
// SxS private-assembly manifests, independently of a copy already in-process.
//
// Scenario:
//   1. Copy openvino.dll from the bin directory to a temp subdirectory.
//   2. Pre-load the copy with DONT_RESOLVE_DLL_REFERENCES — it enters the
//      module list without running DllMain or binding imports (no OV global
//      state is initialised, so there is no conflicting cleanup on unload).
//   3. Register the OVEP plugin.  SxS manifests redirect openvino.dll loads
//      to the assembly directory (the bin directory), causing a second,
//      distinct openvino.dll to be loaded from there.
//   4. Verify that EnumProcessModulesEx reports two separate openvino.dll
//      paths — one from the temp pre-load, one from the bin directory.

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <psapi.h>
#include "ortAbiTestFixture.h"

#include <gtest/gtest.h>

namespace fs = std::filesystem;

namespace {

static fs::path GetBinDir() {
  wchar_t exe_path[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
  return fs::path(exe_path).parent_path();
}

static std::vector<std::wstring> FindLoadedModules(std::wstring_view name_fragment_lower) {
  std::vector<std::wstring> result;
  std::vector<HMODULE> modules(4096);
  DWORD cbNeeded = 0;
  if (!EnumProcessModulesEx(GetCurrentProcess(), modules.data(),
                            static_cast<DWORD>(modules.size() * sizeof(HMODULE)),
                            &cbNeeded, LIST_MODULES_ALL)) {
    return result;
  }
  DWORD count = cbNeeded / sizeof(HMODULE);
  for (DWORD i = 0; i < count; ++i) {
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameExW(GetCurrentProcess(), modules[i], path, MAX_PATH);
    std::wstring wpath(path);
    std::wstring lower = wpath;
    for (auto& c : lower) c = towlower(c);
    if (lower.find(name_fragment_lower) != std::wstring::npos) {
      result.push_back(wpath);
    }
  }
  return result;
}

struct scoped_ep_registration {
  scoped_ep_registration(Ort::Env& env, const char* name, const ORTCHAR_T* path)
      : env_(env), name_(name) {
    env_.RegisterExecutionProviderLibrary(name_, path);
  }
  ~scoped_ep_registration() { env_.UnregisterExecutionProviderLibrary(name_); }

  scoped_ep_registration(const scoped_ep_registration&) = delete;
  scoped_ep_registration& operator=(const scoped_ep_registration&) = delete;
  scoped_ep_registration(scoped_ep_registration&&) = delete;
  scoped_ep_registration& operator=(scoped_ep_registration&&) = delete;

 private:
  Ort::Env& env_;
  const char* name_;
};

}  // namespace

// ---------------------------------------------------------------------------

struct sxs_isolation_fixture : public ortAbiTestFixture {
  fs::path bin_dir_;
  fs::path preloaded_ov_;
  HMODULE h_preloaded_ov_ = nullptr;

  void SetUp() override {
    ResetTempDir();
    bin_dir_ = GetBinDir();
    ASSERT_TRUE(fs::exists(bin_dir_ / L"openvino.dll"))
        << "openvino.dll not found in: " << bin_dir_;

    preloaded_ov_ = GetTempDir() / L"openvino.dll";
    fs::copy_file(bin_dir_ / L"openvino.dll", preloaded_ov_,
                  fs::copy_options::overwrite_existing);

    // DONT_RESOLVE_DLL_REFERENCES maps the PE image into the module list
    // without running DllMain or binding imports — no OV global state is
    // created, so FreeLibrary later causes no conflicting teardown with the
    // SxS-loaded copy.  Normal loading is not possible from an arbitrary
    // directory because the embedded dep manifest requires the openvino_runtime
    // assembly context, which is only present in the bin directory.
    h_preloaded_ov_ = LoadLibraryExW(preloaded_ov_.wstring().c_str(), nullptr,
                                     DONT_RESOLVE_DLL_REFERENCES);
    ASSERT_NE(h_preloaded_ov_, nullptr)
        << "Pre-load of openvino.dll failed (error " << GetLastError() << ")";

    OVEP_TESTS_LOG_INFO << "Preload openvino.dll: " << preloaded_ov_.string();
    LogOvModules();
  }

  void TearDown() override {
    if (h_preloaded_ov_) {
      FreeLibrary(h_preloaded_ov_);
      h_preloaded_ov_ = nullptr;
    }
    fs::remove_all(temp_dir_);
  }

  void LogOvModules() {
    const auto mods = FindLoadedModules(L"openvino.dll");
    OVEP_TESTS_LOG_INFO << "Loaded OpenVINO modules: " << mods.size();
    for (const auto& p : mods) {
      OVEP_TESTS_LOG_INFO << "  " << fs::path(p).string();
    }
  }
};

// ---------------------------------------------------------------------------

TEST_F(sxs_isolation_fixture, sxs_loads_private_assembly_copy) {
  const fs::path plugin_dll = ortAbiTestFixture::GetPluginProviderPath();
  OVEP_TESTS_LOG_INFO << "Register EP from: " << plugin_dll.string();
  // SxS manifests redirect openvino.dll loads to the bin-dir assembly,
  // producing a second in-process instance alongside the pre-loaded copy.
  const scoped_ep_registration ep_reg(env_, "OpenVINOExecutionProvider",
                                      plugin_dll.native().c_str());
  LogOvModules();

  const auto ov_mods = FindLoadedModules(L"openvino.dll");

  ASSERT_GE(ov_mods.size(), 2u) << "SxS isolation failed: expected >=2 openvino.dll instances";

  const auto contains = [&](const fs::path& target) {
    return std::ranges::any_of(ov_mods, [&](const auto& p) {
      std::error_code ec;
      return fs::equivalent(p, target, ec) && !ec;
    });
  };

  EXPECT_TRUE(contains(preloaded_ov_)) << "Pre-loaded openvino.dll not found in module list";
  EXPECT_TRUE(contains(bin_dir_ / L"openvino.dll")) << "SxS bin-dir openvino.dll not found in module list";
}

#endif  // _WIN32
