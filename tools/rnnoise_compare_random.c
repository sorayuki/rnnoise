#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rnnoise.h"

#define FRAME_SIZE 480

static float rand_unit(void) {
  return (float)rand() / (float)RAND_MAX;
}

static void fill_non_silent_frame(float *x, int frame_index) {
  float phase1 = 0.013f * frame_index;
  float phase2 = 0.021f * frame_index;
  int i;
  for (i = 0; i < FRAME_SIZE; ++i) {
    float t = (float)i;
    float tone = 6000.f * sinf(0.031f * t + phase1) + 3500.f * cosf(0.073f * t + phase2);
    float noise = 1800.f * (2.f * rand_unit() - 1.f);
    x[i] = tone + noise;
  }
}

int main(int argc, char **argv) {
  int frames = 64;
  unsigned seed = 1337u;
  int i;
  DenoiseState *st = rnnoise_create(NULL);
  if (st == NULL) {
    fprintf(stderr, "failed to create rnnoise state\n");
    return 1;
  }
  if (argc >= 2) frames = atoi(argv[1]);
  if (argc >= 3) seed = (unsigned)strtoul(argv[2], NULL, 10);
  if (frames <= 0) {
    fprintf(stderr, "frames must be positive\n");
    rnnoise_destroy(st);
    return 1;
  }
  srand(seed);
  for (i = 0; i < frames; ++i) {
    float in[FRAME_SIZE];
    float out[FRAME_SIZE];
    fill_non_silent_frame(in, i);
    rnnoise_process_frame(st, out, in);
  }
  rnnoise_destroy(st);
  fprintf(stderr, "rnnoise_compare_random: processed %d frames with seed %u\n", frames, seed);
  return 0;
}
