#ifndef RNN_WINML_H_
#define RNN_WINML_H_

#include "rnn.h"

int compute_rnn_winml(RNNState *rnn, float *gains, float *vad, const float *input);
void rnn_winml_shutdown(void);

#endif
