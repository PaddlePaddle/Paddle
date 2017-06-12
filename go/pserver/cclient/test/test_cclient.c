#include <stdio.h>
#include <stdlib.h>

#include "libpaddle_pserver_cclient.h"

typedef float real;

void fail() {
  // TODO(helin): fix: gtest using cmake is not working, using this
  // hacky way for now.
  printf("test failed.\n");
  exit(-1);
}

void print_parameter(paddle_gradient* param) {
  if (param == NULL) {
    printf("param is NULL!!\n");
  } else {
    printf("==== parameter ====\n");
    printf("name: %s\n", param->name);
    printf("content_len: %d\n", param->content_len);
    printf("content_type: %d\n", param->element_type);
    int i;
    for (i = 0; i < param->content_len / sizeof(real); ++i) {
      printf("%f ", ((float*)param->content)[i]);
    }
    printf("\n\n");
  }
}

int main() {
  char addr[] = "localhost:3000";
  client c = paddle_new_pserver_client(addr, 1);

  char* names[] = {"param_a", "param_b"};
retry:

  if (paddle_begin_init_params(c)) {
    paddle_parameter param;
    real param_content1[] = {0.1, 0.2, 0.3};
    param.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
    param.name = names[0];
    param.content = (unsigned char*)param_content1;
    param.content_len = 3 * sizeof(real);
    if (paddle_init_param(c, param, NULL, 0) != 0) {
      goto retry;
    }
    real param_content2[] = {0.4, 0.5, 0.6};
    param.element_type = PADDLE_ELEMENT_TYPE_INT32;
    param.name = names[1];
    param.content = (unsigned char*)param_content2;
    param.content_len = 3 * sizeof(real);
    if (paddle_init_param(c, param, NULL, 0) != 0) {
      goto retry;
    }

    if (paddle_finish_init_params(c) != 0) {
      goto retry;
    }
  } else {
    fail();
  }

  printf("get initialized parameters from pserver:\n");
  paddle_parameter* param_ptrs[2] = {NULL, NULL};
  if (paddle_get_params(c, names, param_ptrs, 2) != 0) {
    fail();
  }
  print_parameter(param_ptrs[0]);
  print_parameter(param_ptrs[1]);

  printf("send gradient to pserver:\n");
  real gradient_content1[] = {0.01, 0.02, 0.03};
  real gradinet_content2[] = {0.04, 0.05, 0.06};

  paddle_gradient** grads =
      (paddle_gradient**)malloc(sizeof(paddle_gradient*) * 2);
  grads[0] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  grads[0]->name = names[0];
  grads[0]->content = (unsigned char*)gradient_content1;
  grads[0]->content_len = 3 * sizeof(real);
  grads[0]->element_type = PADDLE_ELEMENT_TYPE_FLOAT32;

  grads[1] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  grads[1]->name = names[1];
  grads[1]->content = (unsigned char*)gradinet_content2;
  grads[1]->content_len = 3 * sizeof(real);
  grads[1]->element_type = PADDLE_ELEMENT_TYPE_INT32;

  print_parameter(grads[0]);
  print_parameter(grads[1]);

  if (paddle_send_grads(c, grads, 2) != 0) {
    fail();
  }

  printf("get updated parameters from pserver:\n");
  // get parameters again by reusing the allocated parameter buffers.
  if (paddle_get_params(c, names, param_ptrs, 2) != 0) {
    fail();
  }

  print_parameter(param_ptrs[0]);
  print_parameter(param_ptrs[1]);

  paddle_release_param(param_ptrs[0]);
  paddle_release_param(param_ptrs[1]);

  if (paddle_save_model(c, "/tmp/") != 0) {
    fail();
  }

  printf("test success!\n");
  return 0;
}
