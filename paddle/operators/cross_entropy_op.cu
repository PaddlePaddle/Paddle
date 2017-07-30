#include "paddle/operators/cross_entropy_op.h"

REGISTER_OP_GPU_KERNEL(onehot_cross_entropy,
                       ops::OnehotCrossEntropyOpKernel<ops::GPUPlace, float>);