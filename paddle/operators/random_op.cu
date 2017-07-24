#include "paddle/operators/random_op.h"
#include "paddle/framework/op_registry.h"

typedef paddle::operators::RandomOpKernel<paddle::platform::GPUPlace, float>
  RandomOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(random_op, RandomOpKernel_GPU_float);