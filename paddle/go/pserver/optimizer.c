#include <stdlib.h>

#include "optimizer.h"

typedef struct {
  double learning_rate;
} SGD_optimizer;

paddle_optimizer* paddle_create_SGD_optimizer(double learning_rate) {
  SGD_optimizer* o = (SGD_optimizer*)malloc(sizeof(SGD_optimizer));
  o->learning_rate = learning_rate;
  return (paddle_optimizer*)o;
}

void paddle_release_optimizer(paddle_optimizer* o) {
  free(o);
}

int paddle_update_parameter(paddle_optimizer* o, void *buffer, paddle_element_type datatype, const void* gradient, int num_bytes) {
  // TODO
  return 0;
}
