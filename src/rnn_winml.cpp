#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <windows.h>

#define ENABLE_NPU_ADAPTER_ENUMERATION 1
#include <WinMLEpCatalog.h>
#include <winml/dml_provider_factory.h>
#include <winml/onnxruntime_c_api.h>
#include <winml/onnxruntime_session_options_config_keys.h>

extern "C" {
#include "rnn_winml.h"
}

namespace {

static const char *input_names[] = {
    "features",
    "conv1_state",
    "conv2_state",
    "gru1_state",
    "gru2_state",
    "gru3_state",
};

static const char *output_names[] = {
    "gains",
    "vad",
    "next_conv1_state",
    "next_conv2_state",
    "next_gru1_state",
    "next_gru2_state",
    "next_gru3_state",
};

struct RegisteredProvider {
  std::string name;
  std::string path;
};

struct CatalogProvider {
  std::string name;
  std::string path;
  WinMLEpReadyState ready_state = WinMLEpReadyState_NotPresent;
  WinMLEpCertification certification = WinMLEpCertification_Unknown;
};

enum class SessionCandidate {
  Npu,
  MaxPerformance,
  DirectMl,
  Cpu,
};

const SessionCandidate kCandidates[] = {
    SessionCandidate::Npu,
    SessionCandidate::MaxPerformance,
    SessionCandidate::DirectMl,
    SessionCandidate::Cpu,
};

struct WinMLState {
  const OrtApi *ort = nullptr;
  OrtEnv *env = nullptr;
  OrtSessionOptions *options = nullptr;
  OrtSession *session = nullptr;
  OrtMemoryInfo *memory_info = nullptr;
  WinMLEpCatalogHandle catalog = nullptr;
  std::vector<RegisteredProvider> providers;
  std::string selected_device;
  std::wstring model_path;
  size_t next_candidate = 0;
  bool ready_openvino_catalog_provider = false;
  bool initialized = false;
};

std::unique_ptr<WinMLState> g_state;
bool g_init_attempted = false;

std::wstring widen(const char *path) {
  if (path == nullptr) return {};
  int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
  if (len == 0) len = MultiByteToWideChar(CP_ACP, 0, path, -1, nullptr, 0);
  if (len == 0) return {};
  std::wstring out(static_cast<size_t>(len - 1), L'\0');
  if (MultiByteToWideChar(CP_UTF8, 0, path, -1, out.data(), len) == 0) {
    MultiByteToWideChar(CP_ACP, 0, path, -1, out.data(), len);
  }
  return out;
}

std::wstring absolute_path(const char *path) {
  std::wstring wpath = widen(path);
  DWORD needed = GetFullPathNameW(wpath.c_str(), 0, nullptr, nullptr);
  if (needed == 0) return wpath;
  std::wstring full(needed, L'\0');
  DWORD written = GetFullPathNameW(wpath.c_str(), needed, full.data(), nullptr);
  if (written == 0) return wpath;
  full.resize(written);
  return full;
}

bool check_status(WinMLState &state, OrtStatus *status, const char *what) {
  if (status == nullptr) return true;
  const char *msg = state.ort != nullptr ? state.ort->GetErrorMessage(status) : "unknown ONNX Runtime error";
  std::fprintf(stderr, "rnnoise winml: %s failed: %s\n", what, msg);
  if (state.ort != nullptr) state.ort->ReleaseStatus(status);
  return false;
}

bool failed_hr(HRESULT hr, const char *what) {
  if (SUCCEEDED(hr)) return false;
  std::fprintf(stderr, "rnnoise winml: %s failed: 0x%08X\n", what, static_cast<unsigned>(hr));
  return true;
}

const char *ready_state_name(WinMLEpReadyState state) {
  switch (state) {
    case WinMLEpReadyState_Ready: return "Ready";
    case WinMLEpReadyState_NotReady: return "NotReady";
    case WinMLEpReadyState_NotPresent: return "NotPresent";
    default: return "Unknown";
  }
}

const char *device_type_name(OrtHardwareDeviceType type) {
  switch (type) {
    case OrtHardwareDeviceType_CPU: return "CPU";
    case OrtHardwareDeviceType_GPU: return "GPU";
    case OrtHardwareDeviceType_NPU: return "NPU";
    default: return "Unknown";
  }
}

bool contains_case_insensitive(const char *text, const char *needle) {
  if (text == nullptr || needle == nullptr) return false;
  std::string lhs(text);
  std::string rhs(needle);
  std::transform(lhs.begin(), lhs.end(), lhs.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  std::transform(rhs.begin(), rhs.end(), rhs.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return lhs.find(rhs) != std::string::npos;
}

void trim_trailing_nulls(std::string &value) {
  while (!value.empty() && value.back() == '\0') value.pop_back();
}

BOOL CALLBACK collect_provider(WinMLEpHandle ep, const WinMLEpInfo *info, void *context) {
  auto *providers = static_cast<std::vector<CatalogProvider> *>(context);
  CatalogProvider provider;
  provider.name = info != nullptr && info->name != nullptr ? info->name : "";
  provider.path = info != nullptr && info->libraryPath != nullptr ? info->libraryPath : "";
  provider.ready_state = info != nullptr ? info->readyState : WinMLEpReadyState_NotPresent;
  provider.certification = info != nullptr ? info->certification : WinMLEpCertification_Unknown;

  if (provider.ready_state == WinMLEpReadyState_NotReady && ep != nullptr) {
    HRESULT hr = WinMLEpEnsureReady(ep);
    if (SUCCEEDED(hr)) {
      WinMLEpReadyState ready = WinMLEpReadyState_NotPresent;
      if (SUCCEEDED(WinMLEpGetReadyState(ep, &ready))) provider.ready_state = ready;
    }
  }
  if (provider.ready_state == WinMLEpReadyState_Ready && ep != nullptr) {
    size_t path_size = 0;
    if (SUCCEEDED(WinMLEpGetLibraryPathSize(ep, &path_size)) && path_size > 0) {
      std::string path(path_size, '\0');
      if (SUCCEEDED(WinMLEpGetLibraryPath(ep, path_size, path.data(), nullptr))) {
        trim_trailing_nulls(path);
        provider.path = path;
      }
    }
  }

  providers->push_back(provider);
  return TRUE;
}

bool register_catalog_providers(WinMLState &state) {
  if (failed_hr(WinMLEpCatalogCreate(&state.catalog), "WinMLEpCatalogCreate")) return true;

  std::vector<CatalogProvider> providers;
  if (failed_hr(WinMLEpCatalogEnumProviders(state.catalog, collect_provider, &providers),
                "WinMLEpCatalogEnumProviders")) {
    return true;
  }

  for (const CatalogProvider &provider : providers) {
    std::fprintf(stderr, "rnnoise winml: catalog provider name=%s ready=%s path=%s\n",
        provider.name.c_str(), ready_state_name(provider.ready_state), provider.path.c_str());
    if (provider.ready_state == WinMLEpReadyState_Ready &&
        contains_case_insensitive(provider.name.c_str(), "openvino")) {
      state.ready_openvino_catalog_provider = true;
    }
    if (provider.ready_state != WinMLEpReadyState_Ready || provider.path.empty()) continue;

    std::string registration_name = provider.name.empty()
        ? "rnnoise_winml_ep_" + std::to_string(state.providers.size())
        : provider.name;
    if (std::any_of(state.providers.begin(), state.providers.end(),
                   [&](const RegisteredProvider &p) { return p.name == registration_name; })) {
      registration_name += "_" + std::to_string(state.providers.size());
    }

    std::wstring library_path = widen(provider.path.c_str());
    if (check_status(state,
        state.ort->RegisterExecutionProviderLibrary(state.env, registration_name.c_str(), library_path.c_str()),
        "RegisterExecutionProviderLibrary")) {
      state.providers.push_back({registration_name, provider.path});
      std::fprintf(stderr, "rnnoise winml: registered provider %s\n", registration_name.c_str());
    }
  }

  if (state.providers.empty()) {
    std::fprintf(stderr, "rnnoise winml: no Windows ML execution providers registered\n");
  }
  return !state.providers.empty();
}

bool create_options(WinMLState &state, bool disable_cpu_fallback) {
  if (state.options != nullptr) {
    state.ort->ReleaseSessionOptions(state.options);
    state.options = nullptr;
  }
  if (!check_status(state, state.ort->CreateSessionOptions(&state.options), "CreateSessionOptions")) return false;
  if (!check_status(state,
      state.ort->SetSessionGraphOptimizationLevel(state.options, ORT_ENABLE_ALL),
      "SetSessionGraphOptimizationLevel")) {
    return false;
  }
  if (disable_cpu_fallback) {
    if (!check_status(state,
        state.ort->AddSessionConfigEntry(state.options, kOrtSessionOptionsDisableCPUEPFallback, "1"),
        "AddSessionConfigEntry(session.disable_cpu_ep_fallback)")) {
      return false;
    }
  }
  return true;
}

std::string describe_ep_device(WinMLState &state, const OrtEpDevice *ep_device) {
  const OrtHardwareDevice *device = state.ort->EpDevice_Device(ep_device);
  OrtHardwareDeviceType type = state.ort->HardwareDevice_Type(device);
  const char *ep_name = state.ort->EpDevice_EpName(ep_device);
  const char *ep_vendor = state.ort->EpDevice_EpVendor(ep_device);
  const char *hw_vendor = state.ort->HardwareDevice_Vendor(device);
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s on %s vendor=%s ep_vendor=%s device_id=%u",
      ep_name != nullptr ? ep_name : "unknown_ep",
      device_type_name(type),
      hw_vendor != nullptr ? hw_vendor : "unknown_hw_vendor",
      ep_vendor != nullptr ? ep_vendor : "unknown_ep_vendor",
      state.ort->HardwareDevice_DeviceId(device));
  return buf;
}

std::vector<const OrtEpDevice *> get_ep_devices(WinMLState &state) {
  const OrtEpDevice *const *ep_devices = nullptr;
  size_t num_ep_devices = 0;
  std::vector<const OrtEpDevice *> out;
  if (!check_status(state, state.ort->GetEpDevices(state.env, &ep_devices, &num_ep_devices), "GetEpDevices")) {
    return out;
  }
  for (size_t i = 0; i < num_ep_devices; ++i) {
    out.push_back(ep_devices[i]);
    std::fprintf(stderr, "rnnoise winml: ep device %zu: %s\n", i, describe_ep_device(state, ep_devices[i]).c_str());
  }
  return out;
}

bool append_ep_device(WinMLState &state, const OrtEpDevice *ep_device) {
  const OrtEpDevice *devices[] = {ep_device};
  return check_status(state,
      state.ort->SessionOptionsAppendExecutionProvider_V2(state.options, state.env, devices, 1, nullptr, nullptr, 0),
      "SessionOptionsAppendExecutionProvider_V2");
}

bool append_openvino_npu(WinMLState &state, const char *device_type) {
  const char *keys[] = {
      "device_type",
      "enable_npu_fast_compile",
  };
  const char *values[] = {
      device_type,
      "1",
  };
  return check_status(state,
      state.ort->SessionOptionsAppendExecutionProvider_OpenVINO_V2(
          state.options, keys, values, sizeof(keys) / sizeof(keys[0])),
      "SessionOptionsAppendExecutionProvider_OpenVINO_V2(NPU)");
}

bool create_session(WinMLState &state, const char *label) {
  if (state.session != nullptr) {
    state.ort->ReleaseSession(state.session);
    state.session = nullptr;
  }
  if (!check_status(state,
      state.ort->CreateSession(state.env, state.model_path.c_str(), state.options, &state.session),
      "CreateSession")) {
    return false;
  }
  state.selected_device = label;
  std::fprintf(stderr, "rnnoise winml: selected device %s\n", state.selected_device.c_str());
  return true;
}

bool has_ready_openvino_provider(const WinMLState &state) {
  if (state.ready_openvino_catalog_provider) return true;
  return std::any_of(state.providers.begin(), state.providers.end(),
      [](const RegisteredProvider &provider) {
        return contains_case_insensitive(provider.name.c_str(), "openvino");
      });
}

bool try_create_openvino_npu_session(WinMLState &state) {
  const char *device_type = std::getenv("RNNOISE_OPENVINO_DEVICE_TYPE");
  if (device_type == nullptr || device_type[0] == '\0') device_type = "NPU";

  if (!has_ready_openvino_provider(state)) {
    std::fprintf(stderr, "rnnoise winml: no registered OpenVINO provider found\n");
    return false;
  }

  if (!create_options(state, true)) return false;
  if (!append_openvino_npu(state, device_type)) return false;

  std::string label = "NPU via OpenVINO device_type=";
  label += device_type;
  return create_session(state, label.c_str());
}

bool try_create_npu_session(WinMLState &state) {
  std::vector<const OrtEpDevice *> ep_devices = get_ep_devices(state);
  for (const OrtEpDevice *ep_device : ep_devices) {
    const OrtHardwareDevice *device = state.ort->EpDevice_Device(ep_device);
    if (state.ort->HardwareDevice_Type(device) != OrtHardwareDeviceType_NPU) continue;
    std::string label = "NPU via " + describe_ep_device(state, ep_device);
    if (!create_options(state, true)) return false;
    if (!append_ep_device(state, ep_device)) continue;
    if (create_session(state, label.c_str())) return true;
  }
  std::fprintf(stderr, "rnnoise winml: no ready catalog NPU EP device found\n");
  if (try_create_openvino_npu_session(state)) return true;

  if (!create_options(state, true)) return false;
  const OrtDmlApi *dml_api = nullptr;
  if (check_status(state,
      state.ort->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&dml_api)),
      "GetExecutionProviderApi(DML)")) {
    OrtDmlDeviceOptions device_options;
    device_options.Preference = HighPerformance;
    device_options.Filter = Npu;
    if (check_status(state,
        dml_api->SessionOptionsAppendExecutionProvider_DML2(state.options, &device_options),
        "SessionOptionsAppendExecutionProvider_DML2(NPU)")) {
      if (create_session(state, "NPU via DirectML DML2 NPU filter")) return true;
    }
  }
  return false;
}

bool try_create_max_performance_session(WinMLState &state) {
  if (!create_options(state, false)) return false;
  if (!check_status(state,
      state.ort->SessionOptionsSetEpSelectionPolicy(state.options, OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE),
      "SessionOptionsSetEpSelectionPolicy(MAX_PERFORMANCE)")) {
    return false;
  }
  return create_session(state, "MAX_PERFORMANCE auto EP selection");
}

bool try_create_directml_session(WinMLState &state) {
  std::vector<const OrtEpDevice *> ep_devices = get_ep_devices(state);
  for (const OrtEpDevice *ep_device : ep_devices) {
    const char *ep_name = state.ort->EpDevice_EpName(ep_device);
    if (!contains_case_insensitive(ep_name, "dml") && !contains_case_insensitive(ep_name, "directml")) continue;
    std::string label = "DirectML via " + describe_ep_device(state, ep_device);
    if (!create_options(state, false)) return false;
    if (!append_ep_device(state, ep_device)) continue;
    if (create_session(state, label.c_str())) return true;
  }
  const OrtDmlApi *dml_api = nullptr;
  if (!create_options(state, false)) return false;
  if (check_status(state,
      state.ort->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void **>(&dml_api)),
      "GetExecutionProviderApi(DML)")) {
    OrtDmlDeviceOptions device_options;
    device_options.Preference = HighPerformance;
    device_options.Filter = Gpu;
    if (check_status(state,
        dml_api->SessionOptionsAppendExecutionProvider_DML2(state.options, &device_options),
        "SessionOptionsAppendExecutionProvider_DML2(GPU)")) {
      if (create_session(state, "DirectML DML2 GPU high performance")) return true;
    }
    if (!create_options(state, false)) return false;
    if (check_status(state,
        dml_api->SessionOptionsAppendExecutionProvider_DML(state.options, 0),
        "SessionOptionsAppendExecutionProvider_DML(0)")) {
      if (create_session(state, "DirectML default adapter")) return true;
    }
  }
  std::fprintf(stderr, "rnnoise winml: no DirectML device found\n");
  return false;
}

bool try_create_cpu_session(WinMLState &state) {
  if (!create_options(state, false)) return false;
  return create_session(state, "CPU");
}

bool select_next_session(WinMLState &state) {
  while (state.next_candidate < sizeof(kCandidates) / sizeof(kCandidates[0])) {
    SessionCandidate candidate = kCandidates[state.next_candidate++];
    switch (candidate) {
      case SessionCandidate::Npu:
        if (try_create_npu_session(state)) return true;
        if (std::getenv("RNNOISE_WINML_REQUIRE_NPU") != nullptr) return false;
        break;
      case SessionCandidate::MaxPerformance:
        if (try_create_max_performance_session(state)) return true;
        break;
      case SessionCandidate::DirectMl:
        if (try_create_directml_session(state)) return true;
        break;
      case SessionCandidate::Cpu:
        if (try_create_cpu_session(state)) return true;
        break;
    }
  }
  return false;
}

int init_winml() {
  if (g_state && g_state->initialized) return 0;
  if (g_init_attempted) return -1;
  g_init_attempted = true;

  auto state = std::make_unique<WinMLState>();
  state->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (state->ort == nullptr) {
    std::fprintf(stderr, "rnnoise winml: failed to get ONNX Runtime API\n");
    return -1;
  }

  const char *model_path = std::getenv("RNNOISE_WINML_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') model_path = "models/rnnoise_rnn.onnx";
  state->model_path = absolute_path(model_path);

  if (!check_status(*state, state->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rnnoise-winml", &state->env),
                    "CreateEnv")) {
    return -1;
  }
  if (!check_status(*state, state->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                                            &state->memory_info),
                    "CreateCpuMemoryInfo")) {
    g_state = std::move(state);
    rnn_winml_shutdown();
    return -1;
  }
  register_catalog_providers(*state);
  if (!select_next_session(*state)) {
    g_state = std::move(state);
    rnn_winml_shutdown();
    return -1;
  }

  std::fwprintf(stderr, L"rnnoise winml: loaded %ls\n", state->model_path.c_str());
  state->initialized = true;
  g_state = std::move(state);
  return 0;
}

bool make_tensor(float *data, size_t count, const int64_t *shape, size_t shape_len, OrtValue **value) {
  return check_status(*g_state,
      g_state->ort->CreateTensorWithDataAsOrtValue(g_state->memory_info, data, count * sizeof(float),
          shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value),
      "CreateTensorWithDataAsOrtValue");
}

bool copy_tensor(OrtValue *value, float *dst, size_t count) {
  float *src = nullptr;
  if (!check_status(*g_state, g_state->ort->GetTensorMutableData(value, reinterpret_cast<void **>(&src)),
                    "GetTensorMutableData")) {
    return false;
  }
  std::memcpy(dst, src, count * sizeof(float));
  return true;
}

}  // namespace

extern "C" int compute_rnn_winml(RNNState *rnn, float *gains, float *vad, const float *input) {
  float features[CONV1_IN_SIZE];
  std::memcpy(features, input, sizeof(features));
  if (init_winml()) return -1;

  while (g_state->session != nullptr) {
    OrtValue *inputs[6] = {0};
    OrtValue *outputs[7] = {0};
    int64_t features_shape[2] = {1, CONV1_IN_SIZE};
    int64_t conv1_state_shape[2] = {1, CONV1_STATE_SIZE};
    int64_t conv2_state_shape[2] = {1, CONV2_STATE_SIZE};
    int64_t gru_state_shape[2] = {1, GRU1_STATE_SIZE};
    int ret = -1;

    if (!make_tensor(features, CONV1_IN_SIZE, features_shape, 2, &inputs[0])) goto done;
    if (!make_tensor(rnn->conv1_state, CONV1_STATE_SIZE, conv1_state_shape, 2, &inputs[1])) goto done;
    if (!make_tensor(rnn->conv2_state, CONV2_STATE_SIZE, conv2_state_shape, 2, &inputs[2])) goto done;
    if (!make_tensor(rnn->gru1_state, GRU1_STATE_SIZE, gru_state_shape, 2, &inputs[3])) goto done;
    if (!make_tensor(rnn->gru2_state, GRU2_STATE_SIZE, gru_state_shape, 2, &inputs[4])) goto done;
    if (!make_tensor(rnn->gru3_state, GRU3_STATE_SIZE, gru_state_shape, 2, &inputs[5])) goto done;

    if (!check_status(*g_state,
        g_state->ort->Run(g_state->session, nullptr, input_names,
            reinterpret_cast<const OrtValue *const *>(inputs), 6, output_names, 7, outputs),
        "Run")) {
      goto done;
    }

    if (!copy_tensor(outputs[0], gains, DENSE_OUT_OUT_SIZE)) goto done;
    if (!copy_tensor(outputs[1], vad, VAD_DENSE_OUT_SIZE)) goto done;
    if (!copy_tensor(outputs[2], rnn->conv1_state, CONV1_STATE_SIZE)) goto done;
    if (!copy_tensor(outputs[3], rnn->conv2_state, CONV2_STATE_SIZE)) goto done;
    if (!copy_tensor(outputs[4], rnn->gru1_state, GRU1_STATE_SIZE)) goto done;
    if (!copy_tensor(outputs[5], rnn->gru2_state, GRU2_STATE_SIZE)) goto done;
    if (!copy_tensor(outputs[6], rnn->gru3_state, GRU3_STATE_SIZE)) goto done;
    ret = 0;

done:
    for (int i = 0; i < 6; ++i) if (inputs[i] != nullptr) g_state->ort->ReleaseValue(inputs[i]);
    for (int i = 0; i < 7; ++i) if (outputs[i] != nullptr) g_state->ort->ReleaseValue(outputs[i]);
    if (ret == 0) return 0;

    std::fprintf(stderr, "rnnoise winml: evaluate on %s failed, trying next device\n",
        g_state->selected_device.c_str());
    if (std::getenv("RNNOISE_WINML_REQUIRE_NPU") != nullptr) break;
    if (!select_next_session(*g_state)) break;
  }

  return -1;
}

extern "C" void rnn_winml_shutdown(void) {
  if (g_state) {
    if (g_state->ort != nullptr) {
      if (g_state->session != nullptr) g_state->ort->ReleaseSession(g_state->session);
      if (g_state->options != nullptr) g_state->ort->ReleaseSessionOptions(g_state->options);
      if (g_state->memory_info != nullptr) g_state->ort->ReleaseMemoryInfo(g_state->memory_info);
      if (g_state->env != nullptr) {
        for (const RegisteredProvider &provider : g_state->providers) {
          OrtStatus *status = g_state->ort->UnregisterExecutionProviderLibrary(g_state->env, provider.name.c_str());
          if (status != nullptr) g_state->ort->ReleaseStatus(status);
        }
        g_state->ort->ReleaseEnv(g_state->env);
      }
    }
    if (g_state->catalog != nullptr) WinMLEpCatalogRelease(g_state->catalog);
  }
  g_state.reset();
  g_init_attempted = false;
}
