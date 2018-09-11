/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>
#include <stdlib.h>

#include "libpaddle_pserver_cclient.h"

// TODO(helin): Fix: gtest using cmake is not working, using this
// hacky way for now.
#define fail()                                          \
  fprintf(stderr, "info: %s:%d: ", __FILE__, __LINE__); \
  exit(-1);

void sendGrads(paddle_pserver_client c) {
  unsigned char grad_a[2000] = {2};
  unsigned char grad_b[3000] = {3};
  paddle_gradient grad1 = {
      "param_a", PADDLE_ELEMENT_TYPE_FLOAT32, grad_a, 2000};
  paddle_gradient grad2 = {
      "param_b", PADDLE_ELEMENT_TYPE_FLOAT32, grad_b, 3000};
  paddle_gradient *grads[2] = {&grad1, &grad2};
  if (paddle_send_grads(c, grads, 2)) {
    fail();
  }
}

void getParams(paddle_pserver_client c) {
  paddle_parameter param_a;
  paddle_parameter param_b;
  char name_a[] = "param_a";
  char name_b[] = "param_b";
  // Must pre-allocate the prameter content before calling paddle_get_params.
  unsigned char content_a[2000] = {};
  unsigned char content_b[3000] = {};
  param_a.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
  param_a.name = name_a;
  param_a.content = content_a;
  param_a.content_len = 2000;
  param_b.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
  param_b.name = name_b;
  param_b.content = content_b;
  param_b.content_len = 3000;

  paddle_parameter *params[2] = {&param_a, &param_b};
  if (paddle_get_params(c, params, 2)) {
    fail();
  }
}

int main() {
  char addr[] = "localhost:3000";
  paddle_pserver_client c = paddle_new_pserver_client(addr, 1);
  char *config_proto;
  size_t config_proto_len = 0;
  ssize_t nread;
  FILE *fp = fopen("testdata/optimizer.pb", "r");
  if (!fp) {
    fail();
  }
  while ((nread = getline(&config_proto, &config_proto_len, fp)) != -1) {
    printf("%s", config_proto);
  }
  fclose(fp);
retry:
  if (paddle_begin_init_params(c)) {
    paddle_parameter param;
    char name_a[] = "param_a";
    char name_b[] = "param_b";
    unsigned char content_a[2000] = {1};
    unsigned char content_b[3000] = {0};
    param.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
    param.name = name_a;
    param.content = content_a;
    param.content_len = 2000;
    int error =
        paddle_init_param(c, param, (void *)config_proto, config_proto_len);
    if (error != 0) {
      goto retry;
    }

    param.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
    param.name = name_b;
    param.content = content_b;
    param.content_len = 3000;
    error = paddle_init_param(c, param, (void *)config_proto, config_proto_len);
    if (error != 0) {
      goto retry;
    }

    error = paddle_finish_init_params(c);
    if (error != 0) {
      goto retry;
    }
  }

  int i;
  for (i = 0; i < 100; i++) {
    sendGrads(c);
    getParams(c);
  }

  return 0;
}
