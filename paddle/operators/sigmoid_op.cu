#include "paddle/operators/sigmoid_op.h"
#include "paddle/framework/op_registry.h"

REGISTER_OP_GPU_KERNEL(
    sigmoid, paddle::operators::SigmoidKernel<paddle::platform::GPUPlace, float>);
