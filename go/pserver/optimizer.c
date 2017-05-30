#include <stdlib.h>

#include "optimizer.h"

typedef int (*update_func)(void*, void*, paddle_element_type, const void*, int);
typedef void (*release_func)(void*);

typedef struct paddle_optimizer {
  update_func update;
  release_func release;
  void* optimizer;
} paddle_optimizer;

void paddle_release_optimizer(paddle_optimizer* o) {
  o->release(o->optimizer);
  free(o);
}

int paddle_update_parameter(paddle_optimizer* o,
                            void* buffer,
                            paddle_element_type element_type,
                            const void* gradient,
                            int num_bytes) {
  return o->update(o->optimizer, buffer, element_type, gradient, num_bytes);
}

typedef struct { double learning_rate; } SGD_optimizer;

int update_SGD(void* optimizer,
               void* buffer,
               paddle_element_type element_type,
               const void* gradient,
               int num_bytes) {
  SGD_optimizer* o = (SGD_optimizer*)optimizer;
  // TODO
  return 0;
}

void release_SGD(void* optimizer) {
  SGD_optimizer* o = (SGD_optimizer*)optimizer;
  // nothing allocated on heap
}

paddle_optimizer* paddle_create_SGD_optimizer(double learning_rate) {
  SGD_optimizer* impl = (SGD_optimizer*)malloc(sizeof(SGD_optimizer));
  impl->learning_rate = learning_rate;
  paddle_optimizer* opt = (paddle_optimizer*)malloc(sizeof(paddle_optimizer));
  opt->update = update_SGD;
  opt->release = release_SGD;
  opt->optimizer = impl;
  return opt;
}
