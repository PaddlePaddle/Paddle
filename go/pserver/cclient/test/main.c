#include <stdio.h>

#include "libclient.h"

void fail() {
  // TODO(helin): fix: gtest using cmake is not working, using this
  // hacky way for now.
  printf("test failed.\n");
  exit(-1);
}

int main() {
  char addr[] = "localhost:3000";
  client c = paddle_new_pserver_client(addr);
retry:
  if (paddle_begin_init_params(c, NULL, 0)) {
    paddle_parameter param;
    char name_a[] = "param_a";
    char name_b[] = "param_b";
    char content[] = {0x00, 0x11, 0x22};
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

  char content[] = {0x00, 0x11, 0x22};
  paddle_gradient grads[2] = {
      {"param_a", PADDLE_ELEMENT_TYPE_INT32, content, 3},
      {"param_b", PADDLE_ELEMENT_TYPE_FLOAT32, content, 3}};

  if (!paddle_send_grads(c, grads, 2)) {
    fail();
  }

  paddle_parameter* params[2] = {NULL, NULL};
  char* names[] = {"param_a", "param_b"};
  if (!paddle_get_params(c, names, params, 2)) {
    fail();
  }

  // get parameters again by reusing the allocated parameter buffers.
  if (!paddle_get_params(c, names, params, 2)) {
    fail();
  }

  paddle_release_param(params[0]);
  paddle_release_param(params[1]);

  if (!paddle_save_model(c, "/tmp/")) {
    fail();
  }

  return 0;
}
