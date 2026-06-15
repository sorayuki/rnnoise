#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "rnn_onnx.h"
#include "onnxruntime_c_api.h"

static const OrtApi *ort;
static OrtEnv *ort_env;
static OrtSessionOptions *ort_options;
static OrtSession *ort_session;
static OrtMemoryInfo *ort_memory_info;
static int ort_init_attempted;

static const char *input_names[] = {
  "features",
  "conv1_state",
  "conv2_state",
  "gru1_state",
  "gru2_state",
  "gru3_state"
};

static const char *output_names[] = {
  "gains",
  "vad",
  "next_conv1_state",
  "next_conv2_state",
  "next_gru1_state",
  "next_gru2_state",
  "next_gru3_state"
};

static int check_status(OrtStatus *status, const char *what) {
  if (status != NULL) {
    const char *msg = ort != NULL ? ort->GetErrorMessage(status) : "unknown ONNX Runtime error";
    fprintf(stderr, "rnnoise onnx: %s failed: %s\n", what, msg);
    if (ort != NULL) ort->ReleaseStatus(status);
    return -1;
  }
  return 0;
}

#ifdef _WIN32
static int make_ort_path(const char *path, wchar_t *wpath, int max_len) {
  int ret = MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, max_len);
  if (ret == 0) ret = MultiByteToWideChar(CP_ACP, 0, path, -1, wpath, max_len);
  return ret != 0 ? 0 : -1;
}
#endif

static int init_onnx(void) {
  const char *model_path;
  OrtStatus *status;
  if (ort_session != NULL) return 0;
  if (ort_init_attempted) return -1;
  ort_init_attempted = 1;
  ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (ort == NULL) {
    fprintf(stderr, "rnnoise onnx: failed to get ONNX Runtime API\n");
    return -1;
  }
  status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rnnoise", &ort_env);
  if (check_status(status, "CreateEnv")) return -1;
  status = ort->CreateSessionOptions(&ort_options);
  if (check_status(status, "CreateSessionOptions")) return -1;
  status = ort->SetSessionGraphOptimizationLevel(ort_options, ORT_DISABLE_ALL);
  if (check_status(status, "SetSessionGraphOptimizationLevel")) return -1;
  model_path = getenv("RNNOISE_ONNX_MODEL");
  if (model_path == NULL || model_path[0] == 0) model_path = "models/rnnoise_rnn.onnx";
#ifdef _WIN32
  {
    wchar_t wpath[1024];
    if (make_ort_path(model_path, wpath, 1024)) {
      fprintf(stderr, "rnnoise onnx: failed to convert model path: %s\n", model_path);
      return -1;
    }
    status = ort->CreateSession(ort_env, wpath, ort_options, &ort_session);
  }
#else
  status = ort->CreateSession(ort_env, model_path, ort_options, &ort_session);
#endif
  if (check_status(status, "CreateSession")) return -1;
  status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info);
  if (check_status(status, "CreateCpuMemoryInfo")) return -1;
  fprintf(stderr, "rnnoise onnx: loaded %s\n", model_path);
  return 0;
}

static int make_tensor(float *data, size_t count, const int64_t *shape, size_t shape_len, OrtValue **value) {
  OrtStatus *status;
  status = ort->CreateTensorWithDataAsOrtValue(ort_memory_info, data, count*sizeof(float),
      shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value);
  return check_status(status, "CreateTensorWithDataAsOrtValue");
}

static int copy_tensor(OrtValue *value, float *dst, size_t count) {
  OrtStatus *status;
  float *src;
  status = ort->GetTensorMutableData(value, (void **)&src);
  if (check_status(status, "GetTensorMutableData")) return -1;
  memcpy(dst, src, count*sizeof(float));
  return 0;
}

int compute_rnn_onnx(RNNState *rnn, float *gains, float *vad, const float *input) {
  OrtValue *inputs[6] = {0};
  OrtValue *outputs[7] = {0};
  float features[CONV1_IN_SIZE];
  int64_t features_shape[2] = {1, CONV1_IN_SIZE};
  int64_t conv1_state_shape[2] = {1, CONV1_STATE_SIZE};
  int64_t conv2_state_shape[2] = {1, CONV2_STATE_SIZE};
  int64_t gru_state_shape[2] = {1, GRU1_STATE_SIZE};
  int ret = -1;
  int i;
  OrtStatus *status;

  if (init_onnx()) return -1;
  memcpy(features, input, sizeof(features));
  if (make_tensor(features, CONV1_IN_SIZE, features_shape, 2, &inputs[0])) goto done;
  if (make_tensor(rnn->conv1_state, CONV1_STATE_SIZE, conv1_state_shape, 2, &inputs[1])) goto done;
  if (make_tensor(rnn->conv2_state, CONV2_STATE_SIZE, conv2_state_shape, 2, &inputs[2])) goto done;
  if (make_tensor(rnn->gru1_state, GRU1_STATE_SIZE, gru_state_shape, 2, &inputs[3])) goto done;
  if (make_tensor(rnn->gru2_state, GRU2_STATE_SIZE, gru_state_shape, 2, &inputs[4])) goto done;
  if (make_tensor(rnn->gru3_state, GRU3_STATE_SIZE, gru_state_shape, 2, &inputs[5])) goto done;
  status = ort->Run(ort_session, NULL, input_names, (const OrtValue * const *)inputs, 6,
      output_names, 7, outputs);
  if (check_status(status, "Run")) goto done;
  if (copy_tensor(outputs[0], gains, DENSE_OUT_OUT_SIZE)) goto done;
  if (copy_tensor(outputs[1], vad, VAD_DENSE_OUT_SIZE)) goto done;
  if (copy_tensor(outputs[2], rnn->conv1_state, CONV1_STATE_SIZE)) goto done;
  if (copy_tensor(outputs[3], rnn->conv2_state, CONV2_STATE_SIZE)) goto done;
  if (copy_tensor(outputs[4], rnn->gru1_state, GRU1_STATE_SIZE)) goto done;
  if (copy_tensor(outputs[5], rnn->gru2_state, GRU2_STATE_SIZE)) goto done;
  if (copy_tensor(outputs[6], rnn->gru3_state, GRU3_STATE_SIZE)) goto done;
  ret = 0;

done:
  for (i=0;i<6;i++) if (inputs[i] != NULL) ort->ReleaseValue(inputs[i]);
  for (i=0;i<7;i++) if (outputs[i] != NULL) ort->ReleaseValue(outputs[i]);
  return ret;
}

void rnn_onnx_shutdown(void) {
  if (ort != NULL) {
    if (ort_memory_info != NULL) ort->ReleaseMemoryInfo(ort_memory_info);
    if (ort_session != NULL) ort->ReleaseSession(ort_session);
    if (ort_options != NULL) ort->ReleaseSessionOptions(ort_options);
    if (ort_env != NULL) ort->ReleaseEnv(ort_env);
  }
  ort_memory_info = NULL;
  ort_session = NULL;
  ort_options = NULL;
  ort_env = NULL;
  ort = NULL;
  ort_init_attempted = 0;
}
