#include "paddle/operators/add_op.h"
#include "paddle/framework/op_registry.h"

REGISTER_OP_GPU_KERNEL(add_two,
                       paddle::operators::AddKernel<paddle::platform::GPUPlace, float>);