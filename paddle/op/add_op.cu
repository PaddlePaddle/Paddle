#include <paddle/op/add_op.h>
#include <paddle/framework/op_registry.h>

REGISTER_OP_GPU_KERNEL(add_two,
                       paddle::op::AddKernel<paddle::platform::GPUPlace>);