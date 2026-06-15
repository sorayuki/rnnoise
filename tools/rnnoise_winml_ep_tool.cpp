#include <cstdio>
#include <cstring>

extern "C" {
#include "rnn_winml.h"
}

static void usage(const char *argv0) {
  std::fprintf(stderr,
      "usage:\n"
      "  %s list\n"
      "  %s ensure <index-or-name-substring>\n",
      argv0, argv0);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
    return rnn_winml_list_execution_providers(nullptr);
  }
  if (argc == 3 && std::strcmp(argv[1], "ensure") == 0) {
    return rnn_winml_list_execution_providers(argv[2]);
  }
  usage(argv[0]);
  return 1;
}