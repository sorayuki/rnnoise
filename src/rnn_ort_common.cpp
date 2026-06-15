#include <cstdio>
#include <cstring>

#include "onnxruntime_c_api.h"

extern "C" {
#include "rnn_ort_common.h"
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

int make_tensor(const RnnOrtRuntime *runtime, float *data, size_t count,
    const int64_t *shape, size_t shape_len, OrtValue **value) {
  return rnn_ort_check_status(runtime,
      runtime->ort->CreateTensorWithDataAsOrtValue(runtime->memory_info, data, count * sizeof(float),
          shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value),
      "CreateTensorWithDataAsOrtValue");
}

int copy_tensor(const RnnOrtRuntime *runtime, OrtValue *value, float *dst, size_t count) {
  float *src = nullptr;
  if (rnn_ort_check_status(runtime,
        runtime->ort->GetTensorMutableData(value, reinterpret_cast<void **>(&src)),
        "GetTensorMutableData")) {
    return -1;
  }
  std::memcpy(dst, src, count * sizeof(float));
  return 0;
}

}  // namespace

extern "C" int rnn_ort_check_status(const RnnOrtRuntime *runtime, OrtStatus *status, const char *what) {
  if (status == nullptr) return 0;
  const char *msg = runtime != nullptr && runtime->ort != nullptr
      ? runtime->ort->GetErrorMessage(status)
      : "unknown ONNX Runtime error";
  const char *prefix = runtime != nullptr && runtime->log_prefix != nullptr ? runtime->log_prefix : "rnnoise ort";
  std::fprintf(stderr, "%s: %s failed: %s\n", prefix, what, msg);
  if (runtime != nullptr && runtime->ort != nullptr) runtime->ort->ReleaseStatus(status);
  return -1;
}

extern "C" int rnn_ort_run_session(
    const RnnOrtRuntime *runtime,
    RNNState *rnn,
    float *gains,
    float *vad,
    const float *input) {
  OrtValue *inputs[6] = {0};
  OrtValue *outputs[7] = {0};
  float features[CONV1_IN_SIZE];
  int64_t features_shape[2] = {1, CONV1_IN_SIZE};
  int64_t conv1_state_shape[2] = {1, CONV1_STATE_SIZE};
  int64_t conv2_state_shape[2] = {1, CONV2_STATE_SIZE};
  int64_t gru_state_shape[2] = {1, GRU1_STATE_SIZE};
  int ret = -1;

  std::memcpy(features, input, sizeof(features));
  if (make_tensor(runtime, features, CONV1_IN_SIZE, features_shape, 2, &inputs[0])) goto done;
  if (make_tensor(runtime, rnn->conv1_state, CONV1_STATE_SIZE, conv1_state_shape, 2, &inputs[1])) goto done;
  if (make_tensor(runtime, rnn->conv2_state, CONV2_STATE_SIZE, conv2_state_shape, 2, &inputs[2])) goto done;
  if (make_tensor(runtime, rnn->gru1_state, GRU1_STATE_SIZE, gru_state_shape, 2, &inputs[3])) goto done;
  if (make_tensor(runtime, rnn->gru2_state, GRU2_STATE_SIZE, gru_state_shape, 2, &inputs[4])) goto done;
  if (make_tensor(runtime, rnn->gru3_state, GRU3_STATE_SIZE, gru_state_shape, 2, &inputs[5])) goto done;

  if (rnn_ort_check_status(runtime,
        runtime->ort->Run(runtime->session, nullptr, input_names,
            reinterpret_cast<const OrtValue *const *>(inputs), 6, output_names, 7, outputs),
        "Run")) {
    goto done;
  }
  if (copy_tensor(runtime, outputs[0], gains, DENSE_OUT_OUT_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[1], vad, VAD_DENSE_OUT_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[2], rnn->conv1_state, CONV1_STATE_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[3], rnn->conv2_state, CONV2_STATE_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[4], rnn->gru1_state, GRU1_STATE_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[5], rnn->gru2_state, GRU2_STATE_SIZE)) goto done;
  if (copy_tensor(runtime, outputs[6], rnn->gru3_state, GRU3_STATE_SIZE)) goto done;
  ret = 0;

done:
  for (int i = 0; i < 6; ++i) if (inputs[i] != nullptr) runtime->ort->ReleaseValue(inputs[i]);
  for (int i = 0; i < 7; ++i) if (outputs[i] != nullptr) runtime->ort->ReleaseValue(outputs[i]);
  return ret;
}
