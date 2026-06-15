#ifndef RNN_ORT_COMMON_H_
#define RNN_ORT_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include "rnn.h"

struct OrtApi;
struct OrtMemoryInfo;
struct OrtSession;
struct OrtStatus;
struct OrtValue;

typedef struct RnnOrtRuntime {
  const OrtApi *ort;
  OrtSession *session;
  OrtMemoryInfo *memory_info;
  const char *log_prefix;
} RnnOrtRuntime;

#ifdef __cplusplus
extern "C" {
#endif

int rnn_ort_check_status(const RnnOrtRuntime *runtime, OrtStatus *status, const char *what);
int rnn_ort_run_session(
    const RnnOrtRuntime *runtime,
    RNNState *rnn,
    float *gains,
    float *vad,
    const float *analysis_window,
    const float *pitch_window,
    int64_t pitch_index);

#ifdef __cplusplus
}
#endif

#endif
