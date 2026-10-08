# Intel.ML.OnnxRuntime.EP.OpenVINO

OpenVINO™ Execution Provider plugin for ONNX Runtime. Enables hardware-accelerated
inference on Intel® CPUs, GPUs, and NPUs.

**Platform support:** Windows x64, Linux x64 (glibc 2.28+, e.g. RHEL 8, Ubuntu 20.04+).

## Usage

```csharp
using Intel.ML.OnnxRuntime.EP.OpenVINO;
using Microsoft.ML.OnnxRuntime;

var env = OrtEnv.Instance();

// Register the plugin EP library
string registrationName = "openvino_ep_registration";
env.RegisterExecutionProviderLibrary(registrationName, OpenVINOEp.GetLibraryPath());

try
{
    // Find the OrtEpDevice for OpenVINO
    OrtEpDevice? epDevice = null;
    foreach (var device in env.GetEpDevices())
    {
        if (string.Equals(OpenVINOEp.GetEpName(), device.EpName, StringComparison.OrdinalIgnoreCase))
        {
            epDevice = device;
            break;
        }
    }

    if (epDevice == null)
        throw new Exception("OpenVINO EP device not found.");

    using var sessionOptions = new SessionOptions();
    sessionOptions.AppendExecutionProvider(env, [epDevice], new Dictionary<string, string>());

    using var session = new InferenceSession("model.onnx", sessionOptions);
    // ... run inference ...
}
finally
{
    env.UnregisterExecutionProviderLibrary(registrationName);
}
```

## Requirements

- [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) 1.23.0+

The OpenVINO runtime libraries (`openvino*.dll`/`libopenvino*.so`, `tbb*.dll`/`libtbb*.so`,
and `libhwloc.so` on Linux) are bundled in this package — no separate OpenVINO installation
is required at runtime.
