#include <stdlib.h>

#include "optimizer.h"

typedef int (*update_func)(void*, void *, paddle_element_type, const void*, int);

typedef struct paddle_optimizer{
  update_func func;
  void* optimizer;
} paddle_optimizer;

void paddle_release_optimizer(paddle_optimizer* o) {
  free(o);
}

int paddle_update_parameter(paddle_optimizer* o, void *buffer, paddle_element_type element_type, const void* gradient, int num_bytes) {
  return o->func(o->optimizer, buffer, element_type, gradient, num_bytes);
}

typedef struct {
  double learning_rate;
} SGD_optimizer;

int paddle_SGD_update_parameter(void* optimizer, void *buffer, paddle_element_type element_type, const void* gradient, int num_bytes) {
  // TODO
  return 0;
}

paddle_optimizer* paddle_create_SGD_optimizer(double learning_rate) {
  SGD_optimizer* o = (SGD_optimizer*)malloc(sizeof(SGD_optimizer));
  o->learning_rate = learning_rate;
  paddle_optimizer* container = (paddle_optimizer*)malloc(sizeof(paddle_optimizer));
  container->func = paddle_SGD_update_parameter;
  container->optimizer = o;
  return container;
}
