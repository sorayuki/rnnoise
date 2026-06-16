#ifndef RNN_ONNX_H_
#define RNN_ONNX_H_

#include "rnn.h"

int compute_rnn_onnx(
    RNNState *rnn,
    float *gains,
    float *vad,
    const float *features);
void rnn_onnx_shutdown(void);

#endif
