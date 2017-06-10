#include <stdio.h>

#include "libpaddle_pserver_cclient.h"

void fail() {
  // TODO(helin): fix: gtest using cmake is not working, using this
  // hacky way for now.
  printf("test failed.\n");
  exit(-1);
}

int main() {
  char addr[] = "localhost:3000";
  client c = paddle_new_pserver_client(addr, 1);
retry:
  if (paddle_begin_init_params(c)) {
    paddle_parameter param;
    char name_a[] = "param_a";
    char name_b[] = "param_b";
    unsigned char content[] = {0x00, 0x11, 0x22};
    param.element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
    param.name = name_a;
    param.content = content;
    param.content_len = 3;
    if (paddle_init_param(c, param, NULL, 0) != 0) {
      goto retry;
    }
    param.element_type = PADDLE_ELEMENT_TYPE_INT32;
    param.name = name_b;
    param.content = content;
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

  unsigned char content[] = {0x00, 0x11, 0x22};
  paddle_gradient** grads =
          (paddle_gradient**)malloc(sizeof(paddle_gradient*) * 2);
  grads[0] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  grads[0]->name = "param_a";
  grads[0]->content = content;
  grads[0]->content_len = 3;
  grads[0]->element_type = PADDLE_ELEMENT_TYPE_FLOAT32;

  grads[1] = (paddle_gradient*)malloc(sizeof(paddle_gradient));
  grads[1]->name = "param_b";
  grads[1]->content = content;
  grads[1]->content_len = 3;
  grads[1]->element_type = PADDLE_ELEMENT_TYPE_INT32;

  if (paddle_send_grads(c, grads, 2) != 0) {
    fail();
  }

  paddle_parameter* params[2] = {NULL, NULL};
  char* names[] = {"param_a", "param_b"};
  if (paddle_get_params(c, names, params, 2) != 0) {
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
