# WinML / ONNX Progress Handoff (2026-06-16)

This note captures the current state of the RNNoise WinML / ONNX work so it can be resumed on another machine.

## What changed

- Moved feature extraction after pitch into the ONNX graph.
- The ONNX model now takes:
  - `analysis_window`
  - `pitch_window`
  - `pitch_index`
  - `conv1_state`
  - `conv2_state`
  - `gru1_state`
  - `gru2_state`
  - `gru3_state`
- Fixed two major model/export issues found during CPU-vs-ONNX comparison:
  - FFT output needed `1 / WINDOW_SIZE` scaling
  - `log10` had accidentally been exported as natural `log`
- Added a WinML EP selector environment variable:
  - `RNNOISE_WINML_EP`

## Current source status

Key touched files:

- `scripts/export_rnn_onnx.py`
- `src/denoise.c`
- `src/denoise.h`
- `src/rnn_onnx.cpp`
- `src/rnn_onnx.h`
- `src/rnn_ort_common.cpp`
- `src/rnn_ort_common.h`
- `src/rnn_winml.cpp`
- `src/rnn_winml.h`
- `CMakeLists.txt`
- `tools/rnnoise_compare_random.c`

## Important runtime controls

### Backend enable

- `RNNOISE_USE_WINML=1`
- `RNNOISE_USE_ONNX=1`

### Model path

- `RNNOISE_WINML_MODEL=D:\devspace\rnnoise\models\rnnoise_rnn.onnx`
- `RNNOISE_ONNX_MODEL=...`

### WinML-specific controls

- `RNNOISE_WINML_REQUIRE_NPU=1`
  - fail instead of falling back if no NPU path succeeds
- `RNNOISE_OPENVINO_DEVICE_TYPE=NPU`
  - only affects the explicit OpenVINO NPU path
- `RNNOISE_WINML_EP=...`
  - newly added selector for forcing a specific WinML execution path
  - accepted values:
    - `auto`
    - `npu`
    - `max`
    - `cpu`
    - `directml` or `dml`
    - `openvino`
    - `tensorrt`, `trt`, or `nvtensorrtx`

## Verified behaviors

### DirectML selector works

Confirmed with:

```powershell
$env:RNNOISE_USE_WINML='1'
$env:RNNOISE_WINML_MODEL='D:\devspace\rnnoise\models\rnnoise_rnn.onnx'
$env:RNNOISE_WINML_EP='directml'
.\build-release-winml-nocompare\Release\rnnoise_demo.exe input.bin output.bin
```

Observed log:

- `rnnoise winml: session selector directml`
- `rnnoise winml: selected device DirectML via DmlExecutionProvider ...`

### WinML provider visibility depends on launch context

When launched in the normal user terminal, the WinML catalog can expose:

- `OpenVINOExecutionProvider`
- `NvTensorRTRTXExecutionProvider`

Earlier agent-launched runs sometimes showed fewer providers. After switching to elevated launch / attach for debugging, the expected providers were visible again. Treat manual terminal runs as the ground truth.

### Full-file WinML run is not a hard hang

Using:

- `build-release-winml-nocompare\Release\input.bin`

The process can appear stuck because:

- `rnnoise_demo` writes output starting from the second frame
- stdio buffering delays visible file growth
- session creation can take a long time before frame output begins

One measured full run finished successfully in roughly 111 seconds.

## Debugging results

### Elevated attach with CDB

Used:

- target: `build-relwithdebinfo-winml-nocompare\RelWithDebInfo\rnnoise_demo.exe`
- debugger:
  - `C:\Program Files\WindowsApps\Microsoft.WinDbg_1.2603.20001.0_x64__8wekyb3d8bbwe\amd64\cdb.exe`

Important finding from the captured stack:

- The process was not stuck in the audio loop.
- It was inside session creation:
  - `onnxruntime!OrtApis::CreateSession`
  - `onnxruntime!InferenceSession::Initialize`
  - `onnxruntime!PluginExecutionProvider::Compile`
  - `tensorrt_rtx_1_4.dll`

Interpretation:

- TensorRT RTX is spending a long time compiling / partitioning / building its execution plan during `CreateSession`.
- This explains the "CPU busy but no output growth" symptom.

### TensorRT warning about `pitch_index`

Observed warning:

```text
Make sure input pitch_index has Int64 binding.
```

The export and runtime binding were checked and are currently consistent:

- model input in ONNX export: `INT64`
- runtime tensor type in `rnn_ort_common.cpp`: `INT64`
- graph casts `pitch_index` to `FLOAT` only after input binding

So this warning currently looks informational, not a confirmed root cause.

## Export / runtime consistency notes

The ONNX graph now computes feature extraction internally from:

- `analysis_window`
- `pitch_window`
- `pitch_index`

The C path still computes the same features locally for CPU/reference use, but external backends are now fed the raw windows plus pitch index.

## Useful commands

### List WinML catalog providers

```powershell
.\build-release-winml-nocompare\Release\rnnoise_winml_ep_tool.exe list
```

### Force DirectML

```powershell
$env:RNNOISE_USE_WINML='1'
$env:RNNOISE_WINML_MODEL='D:\devspace\rnnoise\models\rnnoise_rnn.onnx'
$env:RNNOISE_WINML_EP='directml'
.\build-release-winml-nocompare\Release\rnnoise_demo.exe input.bin output.bin
```

### Force TensorRT RTX

```powershell
$env:RNNOISE_USE_WINML='1'
$env:RNNOISE_WINML_MODEL='D:\devspace\rnnoise\models\rnnoise_rnn.onnx'
$env:RNNOISE_WINML_EP='tensorrt'
.\build-release-winml-nocompare\Release\rnnoise_demo.exe input.bin output.bin
```

### Build RelWithDebInfo WinML no-compare

```powershell
cmake -S . -B build-relwithdebinfo-winml-nocompare -G "Visual Studio 18 2026" -A x64 -DRNNOISE_ENABLE_WINML=ON -DRNNOISE_ENABLE_ONNX=OFF -DRNNOISE_COMPARE_RNN_BACKENDS=OFF
cmake --build build-relwithdebinfo-winml-nocompare --config RelWithDebInfo
```

## Known open issues

1. TensorRT RTX session creation is still very slow and may look like a hang.
2. OpenVINO explicit NPU path still reports missing local `onnxruntime_providers_openvino.dll` if that explicit code path is used.
   - This is separate from the catalog-registered OpenVINO plugin path.
3. No timing log yet around `CreateSession` and first `Run`.
4. CPU-vs-ONNX compare drift is accepted for now, but not fully eliminated.

## Recommended next steps

1. Add timing logs around:
   - `CreateSession`
   - first `Run`
2. Compare:
   - `RNNOISE_WINML_EP=directml`
   - `RNNOISE_WINML_EP=tensorrt`
   using the same input and model
3. If TensorRT RTX remains unusably slow, inspect whether some graph pattern can be simplified for TRT import.
4. Optionally add a progress / flush mechanism to `rnnoise_demo` to make long startup less misleading during manual tests.

## Notes about files not intended for commit

These were local debug artifacts and do not need to be carried forward:

- `cdb_attach.log`
- `directml_probe_stderr.log`
- `rnnoise_relwithdebinfo_stderr.log`
- `tmp_winml_stderr.txt`
- `tmp_winml_stdout.txt`
- `scripts/__pycache__/`

