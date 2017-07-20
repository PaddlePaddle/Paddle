#include "paddle/operators/cross_entropy_op.h"
#include "paddle/framework/op_registry.h"

REGISTER_OP_GPU_KERNEL(onehot_cross_entropy,
                       paddle::operators::OnehotCrossEntropyOpKernel<
                            ::paddle::platform::GPUPlace, float>);