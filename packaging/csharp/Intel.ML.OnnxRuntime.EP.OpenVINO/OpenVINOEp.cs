// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Intel.ML.OnnxRuntime.EP.OpenVINO;

public static class OpenVINOEp
{
    /// <summary>
    /// Returns the path to the OpenVINO EP plugin library contained by this package.
    /// Pass this path to <c>OrtEnv.RegisterExecutionProviderLibrary()</c>.
    /// </summary>
    /// <remarks>
    /// This method resolves the native library relative to the assembly's output directory,
    /// where the package's <c>runtimes/&lt;rid&gt;/native/</c> files are copied at build time
    /// via <c>CopyToOutputDirectory</c>. It does not probe the NuGet package cache directly,
    /// so the consuming project must not suppress native file copying (e.g. via
    /// <c>ExcludeAssets="runtime"</c>).
    /// </remarks>
    /// <returns>Full path to the EP library.</returns>
    /// <exception cref="FileNotFoundException">If the EP library is not found.</exception>
    /// <exception cref="PlatformNotSupportedException">If the current platform is not supported.</exception>
    public static string GetLibraryPath()
    {
        string osArch;
        string libName;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) &&
            RuntimeInformation.OSArchitecture == Architecture.X64)
        {
            osArch = "win-x64";
            libName = "onnxruntime_providers_openvino_plugin.dll";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
                 RuntimeInformation.OSArchitecture == Architecture.X64)
        {
            osArch = "linux-x64";
            libName = "libonnxruntime_providers_openvino_plugin.so";
        }
        else
        {
            throw new PlatformNotSupportedException(
                $"Intel.ML.OnnxRuntime.EP.OpenVINO does not support " +
                $"{RuntimeInformation.OSDescription} {RuntimeInformation.OSArchitecture}. " +
                $"Supported platforms: Windows x64, Linux x64.");
        }

        string rootDir = GetNativeDirectory();
        string epLibPath = Path.GetFullPath(
            Path.Combine(rootDir, "runtimes", osArch, "native", libName));

        if (!File.Exists(epLibPath))
        {
            throw new FileNotFoundException(
                $"OpenVINO EP library not found: {epLibPath}. The package may be corrupted.",
                epLibPath);
        }

        return epLibPath;
    }

    /// <summary>
    /// Returns the names of the EPs created by the plugin library.
    /// Use these names to select an <c>OrtEpDevice</c> from <c>OrtEnv.GetEpDevices()</c>.
    /// </summary>
    /// <returns>Array of EP names.</returns>
    public static string[] GetEpNames()
    {
        return new[] { "OpenVINOExecutionProvider" };
    }

    /// <summary>
    /// Returns the name of the single EP provided by this plugin library.
    /// </summary>
    /// <returns>EP name.</returns>
    public static string GetEpName()
    {
        return GetEpNames()[0];
    }

    private static string GetNativeDirectory()
    {
        var assemblyDir = Path.GetDirectoryName(typeof(OpenVINOEp).Assembly.Location);

        // Works for framework-dependent deployments
        if (!string.IsNullOrEmpty(assemblyDir) && Directory.Exists(assemblyDir))
            return assemblyDir;

        // Fallback for single-file / self-contained apps
        return AppContext.BaseDirectory;
    }
}
