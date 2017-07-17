#include <paddle/operators/adam_op.h>
#include <paddle/framework/op_registry.h>

REGISTER_OP_GPU_KERNEL(momentum_sgd_op,
                       paddle::operators::MomentumSGDOpKernel<paddle::platform::GPUPlace>);