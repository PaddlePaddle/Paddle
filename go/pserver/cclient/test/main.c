#include <stdio.h>
#include <stdlib.h>

#include "libpaddle_pserver_cclient.h"

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
    for (int i = 0; i < param->content_len; ++i) {
      printf("0x%x ", param->content[i]);
    }
    printf("\n\n");
  }
}

int main() {
  char addr[] = "localhost:3000";
  client c = paddle_new_pserver_client(addr, 1);
retry:
  if (paddle_begin_init_params(c)) {
    paddle_parameter param;
    char name_a[] = "param_a";
    char name_b[] = "param_b";
    unsigned char content1[] = {0x01, 0x02, 0x03};
    param.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
    param.name = name_a;
    param.content = content1;
    param.content_len = 3;
    if (paddle_init_param(c, param, NULL, 0) != 0) {
      goto retry;
    }
    unsigned char content2[] = {0x04, 0x05, 0x06};
    param.element_type = PADDLE_ELEMENT_TYPE_INT32;
    param.name = name_b;
    param.content = content2;
    param.content_len = 3;
    if (paddle_init_param(c, param, NULL, 0) != 0) {
      goto retry;
    }

    if (paddle_finish_init_params(c) != 0) {
      goto retry;
    }
  } else {
    fail();
  }

  unsigned char content1[] = {0x12, 0x23, 0x34};
  unsigned char content2[] = {0x45, 0x56, 0x67};

  paddle_gradient** new_params =
      (paddle_gradient**)malloc(sizeof(paddle_gradient*) * 2);
  new_params[0] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  new_params[0]->name = "param_a";
  new_params[0]->content = content1;
  new_params[0]->content_len = 3;
  new_params[0]->element_type = PADDLE_ELEMENT_TYPE_FLOAT32;

  new_params[1] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  new_params[1]->name = "param_b";
  new_params[1]->content = content2;
  new_params[1]->content_len = 3;
  new_params[1]->element_type = PADDLE_ELEMENT_TYPE_INT32;

  print_parameter(new_params[0]);
  print_parameter(new_params[1]);

  if (paddle_send_grads(c, new_params, 2) != 0) {
    fail();
  }

  paddle_parameter* params[2] = {NULL, NULL};
  char* names[] = {"param_a", "param_b"};
  if (paddle_get_params(c, names, params, 2) != 0) {
    fail();
  }

  print_parameter(params[0]);
  print_parameter(params[1]);

  /// change name of parameter.
  char* names2[] = {"param_1", "param_2"};
  if (paddle_get_params(c, names2, params, 2) == 0) {
    fail();
  }

  // get parameters again by reusing the allocated parameter buffers.
  if (paddle_get_params(c, names, params, 2) != 0) {
    fail();
  }

  paddle_release_param(params[0]);
  paddle_release_param(params[1]);

  if (paddle_save_model(c, "/tmp/") != 0) {
    fail();
  }

  printf("test success!\n");
  return 0;
}
