#include "paddle/framework/op_registry.h"
#include "paddle/operators/rowwise_add_op.h"

REGISTER_OP_GPU_KERNEL(
    rowwise_add,
    paddle::operators::RowWiseAddKernel<paddle::platform ::GPUPlace, float>);
