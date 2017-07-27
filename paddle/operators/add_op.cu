#include "paddle/framework/op_registry.h"
#include "paddle/operators/add_op.h"

REGISTER_OP_GPU_KERNEL(add_two, ops::AddKernel<ops::GPUPlace, float>);
