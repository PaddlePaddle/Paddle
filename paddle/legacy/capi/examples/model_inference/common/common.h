//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
