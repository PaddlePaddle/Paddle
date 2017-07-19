#include "paddle/framework/op_registry.h"
#include "paddle/operators/softmax_op.h"

REGISTER_OP_GPU_KERNEL(
    softmax, paddle::operators::SoftmaxKernel<paddle::platform::GPUPlace, float>);
