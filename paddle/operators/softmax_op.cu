#define EIGEN_USE_GPU
#include "paddle/framework/op_registry.h"
#include "paddle/operators/softmax_op.h"

REGISTER_OP_GPU_KERNEL(softmax, ops::SoftmaxKernel<ops::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(softmax_grad,
                       ops::SoftmaxGradKernel<ops::GPUPlace, float>);
