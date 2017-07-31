#include "paddle/framework/op_registry.h"
#include "paddle/operators/mean_op.h"

REGISTER_OP_GPU_KERNEL(mean, ops::AddKernel<ops::GPUPlace, float>);
