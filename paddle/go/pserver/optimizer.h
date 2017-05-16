#ifndef PADDLE_PSERVER_OPTIMIZER_H
#define PADDLE_PSERVER_OPTIMIZER_H

typedef enum {
  PADDLE_ELEMENT_TYPE_INT32   = 0,
  PADDLE_ELEMENT_TYPE_UINT32  = 1,
  PADDLE_ELEMENT_TYPE_INT64   = 2,
  PADDLE_ELEMENT_TYPE_UINT64  = 3,
  PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
  PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
} paddle_element_type;

typedef struct paddle_optimizer paddle_optimizer;

paddle_optimizer* paddle_create_SGD_optimizer(double learning_rate);
void paddle_release_optimizer(paddle_optimizer* o);
int paddle_update_parameter(paddle_optimizer* o, void *buffer, paddle_element_type datatype, const void* gradient, int num_bytes);

#endif /* PADDLE_PSERVER_OPTIMIZER_H */
