#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

#include "onnxruntime_c_api.h"

extern "C" {
#include "rnn_onnx.h"
#include "rnn_ort_common.h"
}

namespace {

struct OnnxState {
  const OrtApi *ort = nullptr;
  OrtEnv *env = nullptr;
  OrtSessionOptions *options = nullptr;
  OrtSession *session = nullptr;
  OrtMemoryInfo *memory_info = nullptr;
  bool init_attempted = false;
};

OnnxState g_state;

#ifdef _WIN32
int make_ort_path(const char *path, wchar_t *wpath, int max_len) {
  int ret = MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, max_len);
  if (ret == 0) ret = MultiByteToWideChar(CP_ACP, 0, path, -1, wpath, max_len);
  return ret != 0 ? 0 : -1;
}
#endif

RnnOrtRuntime runtime_view() {
  return {g_state.ort, g_state.session, g_state.memory_info, "rnnoise onnx"};
}

int init_onnx() {
  const char *model_path = nullptr;
  OrtStatus *status = nullptr;
  if (g_state.session != nullptr) return 0;
  if (g_state.init_attempted) return -1;
  g_state.init_attempted = true;

  g_state.ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (g_state.ort == nullptr) {
    std::fprintf(stderr, "rnnoise onnx: failed to get ONNX Runtime API\n");
    return -1;
  }

  RnnOrtRuntime runtime = runtime_view();
  status = g_state.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rnnoise", &g_state.env);
  if (rnn_ort_check_status(&runtime, status, "CreateEnv")) return -1;
  status = g_state.ort->CreateSessionOptions(&g_state.options);
  if (rnn_ort_check_status(&runtime, status, "CreateSessionOptions")) return -1;
  status = g_state.ort->SetSessionGraphOptimizationLevel(g_state.options, ORT_DISABLE_ALL);
  if (rnn_ort_check_status(&runtime, status, "SetSessionGraphOptimizationLevel")) return -1;

  model_path = std::getenv("RNNOISE_ONNX_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') model_path = "models/rnnoise_rnn.onnx";
#ifdef _WIN32
  {
    wchar_t wpath[1024];
    if (make_ort_path(model_path, wpath, 1024)) {
      std::fprintf(stderr, "rnnoise onnx: failed to convert model path: %s\n", model_path);
      return -1;
    }
    status = g_state.ort->CreateSession(g_state.env, wpath, g_state.options, &g_state.session);
  }
#else
  status = g_state.ort->CreateSession(g_state.env, model_path, g_state.options, &g_state.session);
#endif
  if (rnn_ort_check_status(&runtime, status, "CreateSession")) return -1;
  status = g_state.ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_state.memory_info);
  if (rnn_ort_check_status(&runtime, status, "CreateCpuMemoryInfo")) return -1;
  std::fprintf(stderr, "rnnoise onnx: loaded %s\n", model_path);
  return 0;
}

}  // namespace

extern "C" int compute_rnn_onnx(RNNState *rnn, float *gains, float *vad, const float *input) {
  if (init_onnx()) return -1;
  RnnOrtRuntime runtime = runtime_view();
  return rnn_ort_run_session(&runtime, rnn, gains, vad, input);
}

extern "C" void rnn_onnx_shutdown(void) {
  if (g_state.ort != nullptr) {
    if (g_state.memory_info != nullptr) g_state.ort->ReleaseMemoryInfo(g_state.memory_info);
    if (g_state.session != nullptr) g_state.ort->ReleaseSession(g_state.session);
    if (g_state.options != nullptr) g_state.ort->ReleaseSessionOptions(g_state.options);
    if (g_state.env != nullptr) g_state.ort->ReleaseEnv(g_state.env);
  }
  g_state.memory_info = nullptr;
  g_state.session = nullptr;
  g_state.options = nullptr;
  g_state.env = nullptr;
  g_state.ort = nullptr;
  g_state.init_attempted = false;
}
