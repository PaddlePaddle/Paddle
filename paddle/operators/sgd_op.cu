#include "paddle/operators/sgd_op.h"
#include "paddle/framework/op_registry.h"

typedef paddle::operators::SGDOpKernel<::paddle::platform::GPUPlace, float> SGDOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(sgd, SGDOpKernel_GPU_float);