# Packaging

Distribution packaging for the OpenVINO™ Execution Provider plugin. Each subdirectory produces one package format from the prebuilt binaries.

## Subdirectories

| Directory | Format | Platform | CMake option |
|-----------|--------|----------|--------------|
| `python/` | Python wheel (`.whl`) | Windows, Linux | `-DOVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON` |
| `csharp/` | NuGet package (`.nupkg`) | Windows, Linux | `-DOVEP_PLUGIN_BUILD_NUGET=ON` |
| `winml/` | Loose MSIX package | Windows only | `-DOVEP_PLUGIN_BUILD_MSIX=ON` |

## How it works

None of the packaging targets are built by default. Each must be explicitly enabled via its CMake option at configure time. They each glob binaries from the install-prefix binary directory (`$<CONFIG>/bin` on Windows, `$<CONFIG>/lib` on Linux) using the shared `OVEP_PLUGIN_INSTALL_GLOB_PATTERNS` defined in `CMakeLists.txt`.

```bash
# Windows
cmake --preset windows-x64 -DOVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON -DOVEP_PLUGIN_BUILD_NUGET=ON -DOVEP_PLUGIN_BUILD_MSIX=ON
cmake --build --preset windows-x64 --config RelWithDebInfo --parallel

# Linux
cmake --preset linux-x64-relwithdebinfo -DOVEP_PLUGIN_BUILD_PYTHON_WHEEL=ON -DOVEP_PLUGIN_BUILD_NUGET=ON
cmake --build --preset linux-x64-relwithdebinfo --parallel
```

Packages are written to:

| Format | Output path |
|--------|-------------|
| Python wheel | `<install>/<config>/python/dist/` |
| NuGet | `<install>/<config>/csharp/nuget/` |
| MSIX | `<install>/<config>/winml/msix/` |

## Registering the MSIX loose package

```powershell
Add-AppxPackage -Register "<install>\<config>\winml\msix\AppxManifest.xml"
```
