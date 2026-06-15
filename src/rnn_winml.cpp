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
#include "onnxruntime_c_api.h"
#include <winml/onnxruntime_session_options_config_keys.h>

extern "C" {
#include "rnn_ort_common.h"
#include "rnn_winml.h"
}

namespace {

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
  Cpu,
};

enum class ForcedSessionKind {
  Auto,
  Npu,
  MaxPerformance,
  Cpu,
  Dml,
  OpenVino,
  NvTensorRtx,
};

const SessionCandidate kCandidates[] = {
    SessionCandidate::Npu,
    SessionCandidate::MaxPerformance,
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
  RnnOrtRuntime runtime = {state.ort, state.session, state.memory_info, "rnnoise winml"};
  return rnn_ort_check_status(&runtime, status, what) == 0;
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

std::string format_hresult(HRESULT hr) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08X", static_cast<unsigned>(hr));
  return buf;
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

ForcedSessionKind parse_forced_session_kind(const char *value) {
  if (value == nullptr || value[0] == '\0') return ForcedSessionKind::Auto;
  if (contains_case_insensitive(value, "auto")) return ForcedSessionKind::Auto;
  if (contains_case_insensitive(value, "npu")) return ForcedSessionKind::Npu;
  if (contains_case_insensitive(value, "max")) return ForcedSessionKind::MaxPerformance;
  if (contains_case_insensitive(value, "cpu")) return ForcedSessionKind::Cpu;
  if (contains_case_insensitive(value, "dml") || contains_case_insensitive(value, "directml")) {
    return ForcedSessionKind::Dml;
  }
  if (contains_case_insensitive(value, "openvino")) return ForcedSessionKind::OpenVino;
  if (contains_case_insensitive(value, "nvtensor") ||
      contains_case_insensitive(value, "trt") ||
      contains_case_insensitive(value, "tensorrt")) {
    return ForcedSessionKind::NvTensorRtx;
  }
  std::fprintf(stderr, "rnnoise winml: unknown RNNOISE_WINML_EP=%s, falling back to auto\n", value);
  return ForcedSessionKind::Auto;
}

const char *forced_session_kind_name(ForcedSessionKind kind) {
  switch (kind) {
    case ForcedSessionKind::Auto: return "auto";
    case ForcedSessionKind::Npu: return "npu";
    case ForcedSessionKind::MaxPerformance: return "max_performance";
    case ForcedSessionKind::Cpu: return "cpu";
    case ForcedSessionKind::Dml: return "directml";
    case ForcedSessionKind::OpenVino: return "openvino";
    case ForcedSessionKind::NvTensorRtx: return "nvtensorrtx";
    default: return "unknown";
  }
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

  if (provider.ready_state != WinMLEpReadyState_Ready && ep != nullptr) {
    std::fprintf(stderr, "rnnoise winml: ensuring catalog provider %s ready from %s\n",
        provider.name.c_str(), ready_state_name(provider.ready_state));
    HRESULT hr = WinMLEpEnsureReady(ep);
    if (SUCCEEDED(hr)) {
      WinMLEpReadyState ready = WinMLEpReadyState_NotPresent;
      hr = WinMLEpGetReadyState(ep, &ready);
      if (SUCCEEDED(hr)) {
        provider.ready_state = ready;
      } else {
        std::fprintf(stderr, "rnnoise winml: WinMLEpGetReadyState(%s) failed after EnsureReady: %s\n",
            provider.name.c_str(), format_hresult(hr).c_str());
      }
    } else {
      std::fprintf(stderr, "rnnoise winml: WinMLEpEnsureReady(%s) failed: %s\n",
          provider.name.c_str(), format_hresult(hr).c_str());
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

bool ep_name_contains(WinMLState &state, const OrtEpDevice *ep_device, const char *needle) {
  const char *ep_name = state.ort->EpDevice_EpName(ep_device);
  return contains_case_insensitive(ep_name, needle);
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

bool try_create_cpu_session(WinMLState &state) {
  if (!create_options(state, false)) return false;
  return create_session(state, "CPU");
}

bool select_forced_session(WinMLState &state, ForcedSessionKind forced_kind);

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
      case SessionCandidate::Cpu:
        if (try_create_cpu_session(state)) return true;
        break;
    }
  }
  return false;
}

bool try_create_named_gpu_session(WinMLState &state, const char *ep_name_substring, const char *label_prefix) {
  std::vector<const OrtEpDevice *> ep_devices = get_ep_devices(state);
  for (const OrtEpDevice *ep_device : ep_devices) {
    const OrtHardwareDevice *device = state.ort->EpDevice_Device(ep_device);
    if (state.ort->HardwareDevice_Type(device) != OrtHardwareDeviceType_GPU) continue;
    if (!ep_name_contains(state, ep_device, ep_name_substring)) continue;
    std::string label = std::string(label_prefix) + describe_ep_device(state, ep_device);
    if (!create_options(state, true)) return false;
    if (!append_ep_device(state, ep_device)) continue;
    if (create_session(state, label.c_str())) return true;
  }
  std::fprintf(stderr, "rnnoise winml: no GPU EP device matched %s\n", ep_name_substring);
  return false;
}

bool try_create_dml_session(WinMLState &state) {
  return try_create_named_gpu_session(state, "DmlExecutionProvider", "DirectML via ");
}

bool try_create_openvino_session(WinMLState &state) {
  std::vector<const OrtEpDevice *> ep_devices = get_ep_devices(state);
  for (const OrtEpDevice *ep_device : ep_devices) {
    if (!ep_name_contains(state, ep_device, "OpenVINOExecutionProvider")) continue;
    std::string label = "OpenVINO via " + describe_ep_device(state, ep_device);
    if (!create_options(state, true)) return false;
    if (!append_ep_device(state, ep_device)) continue;
    if (create_session(state, label.c_str())) return true;
  }
  std::fprintf(stderr, "rnnoise winml: no EP device matched OpenVINOExecutionProvider\n");
  return false;
}

bool try_create_nv_tensorrt_rtx_session(WinMLState &state) {
  return try_create_named_gpu_session(state, "NvTensorRTRTXExecutionProvider", "NvTensorRTRTX via ");
}

bool select_forced_session(WinMLState &state, ForcedSessionKind forced_kind) {
  switch (forced_kind) {
    case ForcedSessionKind::Auto:
      return select_next_session(state);
    case ForcedSessionKind::Npu:
      return try_create_npu_session(state);
    case ForcedSessionKind::MaxPerformance:
      return try_create_max_performance_session(state);
    case ForcedSessionKind::Cpu:
      return try_create_cpu_session(state);
    case ForcedSessionKind::Dml:
      return try_create_dml_session(state);
    case ForcedSessionKind::OpenVino:
      return try_create_openvino_session(state);
    case ForcedSessionKind::NvTensorRtx:
      return try_create_nv_tensorrt_rtx_session(state);
    default:
      return false;
  }
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
  const char *forced_ep = std::getenv("RNNOISE_WINML_EP");
  ForcedSessionKind forced_kind = parse_forced_session_kind(forced_ep);
  std::fprintf(stderr, "rnnoise winml: session selector %s\n", forced_session_kind_name(forced_kind));

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
  if (!select_forced_session(*state, forced_kind)) {
    g_state = std::move(state);
    rnn_winml_shutdown();
    return -1;
  }

  std::fwprintf(stderr, L"rnnoise winml: loaded %ls\n", state->model_path.c_str());
  state->initialized = true;
  g_state = std::move(state);
  return 0;
}

}  // namespace

extern "C" int compute_rnn_winml(
    RNNState *rnn,
    float *gains,
    float *vad,
    const float *analysis_window,
    const float *pitch_window,
    int pitch_index) {
  if (init_winml()) return -1;

  while (g_state->session != nullptr) {
    RnnOrtRuntime runtime = {g_state->ort, g_state->session, g_state->memory_info, "rnnoise winml"};
    int ret = rnn_ort_run_session(&runtime, rnn, gains, vad, analysis_window, pitch_window, pitch_index);
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

struct ListCatalogContext {
  const char *ensure_selector = nullptr;
  size_t index = 0;
  bool matched = false;
};

BOOL CALLBACK list_catalog_provider(WinMLEpHandle ep, const WinMLEpInfo *info, void *context) {
  auto *ctx = static_cast<ListCatalogContext *>(context);
  const char *name = info != nullptr && info->name != nullptr ? info->name : "";
  const char *path = info != nullptr && info->libraryPath != nullptr ? info->libraryPath : "";
  WinMLEpReadyState ready = info != nullptr ? info->readyState : WinMLEpReadyState_NotPresent;

  std::printf("[%zu] name=%s ready=%s path=%s\n",
      ctx->index,
      name[0] != '\0' ? name : "<unknown>",
      ready_state_name(ready),
      path[0] != '\0' ? path : "<none>");

  if (ctx->ensure_selector != nullptr && ctx->ensure_selector[0] != '\0') {
    bool should_ensure = false;
    char *end = nullptr;
    unsigned long selected_index = std::strtoul(ctx->ensure_selector, &end, 10);
    if (end != ctx->ensure_selector && *end == '\0') {
      should_ensure = ctx->index == static_cast<size_t>(selected_index);
    } else {
      should_ensure = contains_case_insensitive(name, ctx->ensure_selector) ||
          contains_case_insensitive(path, ctx->ensure_selector);
    }
    if (should_ensure) {
      ctx->matched = true;
      if (ep != nullptr) {
        std::fprintf(stderr, "rnnoise winml: ensuring provider [%zu] %s\n", ctx->index, name);
        HRESULT hr = WinMLEpEnsureReady(ep);
        if (SUCCEEDED(hr)) {
          WinMLEpReadyState updated = WinMLEpReadyState_NotPresent;
          if (SUCCEEDED(WinMLEpGetReadyState(ep, &updated))) {
            std::printf("[%zu] after ensure ready=%s\n", ctx->index, ready_state_name(updated));
          }
        } else {
          std::fprintf(stderr, "rnnoise winml: WinMLEpEnsureReady(%s) failed: %s\n",
              name, format_hresult(hr).c_str());
        }
      }
    }
  }

  ctx->index++;
  return TRUE;
}

extern "C" int rnn_winml_list_execution_providers(const char *ensure_selector) {
  WinMLEpCatalogHandle catalog = nullptr;
  HRESULT hr = WinMLEpCatalogCreate(&catalog);
  if (FAILED(hr)) {
    std::fprintf(stderr, "rnnoise winml: WinMLEpCatalogCreate failed: %s\n", format_hresult(hr).c_str());
    return 1;
  }

  std::printf("Windows ML catalog execution providers:\n");
  ListCatalogContext ctx;
  ctx.ensure_selector = ensure_selector;
  hr = WinMLEpCatalogEnumProviders(catalog, list_catalog_provider, &ctx);
  WinMLEpCatalogRelease(catalog);
  if (FAILED(hr)) {
    std::fprintf(stderr, "rnnoise winml: WinMLEpCatalogEnumProviders failed: %s\n", format_hresult(hr).c_str());
    return 1;
  }
  if (ensure_selector != nullptr && ensure_selector[0] != '\0' && !ctx.matched) {
    std::fprintf(stderr, "rnnoise winml: no provider matched selector: %s\n", ensure_selector);
    return 2;
  }
  return 0;
}
