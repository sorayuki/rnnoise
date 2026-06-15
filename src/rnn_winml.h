#ifndef RNN_WINML_H_
#define RNN_WINML_H_

#include "rnn.h"

int compute_rnn_winml(
    RNNState *rnn,
    float *gains,
    float *vad,
    const float *analysis_window,
    const float *pitch_window,
    int pitch_index);
void rnn_winml_shutdown(void);
int rnn_winml_list_execution_providers(const char *ensure_selector);

#endif
