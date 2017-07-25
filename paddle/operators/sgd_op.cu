#include "paddle/operators/sgd_op.h"

REGISTER_OP_GPU_KERNEL(sgd, ops::SGDOpKernel<ops::GPUPlace, float>);