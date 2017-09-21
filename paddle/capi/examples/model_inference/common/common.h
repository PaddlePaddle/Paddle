#ifndef __CAPI_EXAMPLE_COMMON_H__
#define __CAPI_EXAMPLE_COMMON_H__
#include <stdio.h>
#include <stdlib.h>

#define CHECK(stmt)                                                      \
  do {                                                                   \
    paddle_error __err__ = stmt;                                         \
    if (__err__ != kPD_NO_ERROR) {                                       \
      fprintf(stderr, "Invoke paddle error %d in " #stmt "\n", __err__); \
      exit(__err__);                                                     \
    }                                                                    \
  } while (0)

void* read_config(const char* filename, long* size) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Open %s error\n", filename);
    return NULL;
  }
  fseek(file, 0L, SEEK_END);
  *size = ftell(file);
  fseek(file, 0L, SEEK_SET);
  void* buf = malloc(*size);
  fread(buf, 1, *size, file);
  fclose(file);
  return buf;
}
#endif
