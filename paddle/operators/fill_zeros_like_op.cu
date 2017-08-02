#include "paddle/framework/op_registry.h"
#include "paddle/operators/fill_zeros_like_op.h"

REGISTER_OP_GPU_KERNEL(
    fill_zeros_like,
    paddle::operators::FillZerosLikeKernel<paddle::platform::GPUPlace, float>);