#include "paddle/operators/sigmoid_op.h"

REGISTER_OP_GPU_KERNEL(
    sigmoid, paddle::operators::FakeKernel<paddle::platform::GPUPlace>);
