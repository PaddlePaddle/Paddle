#define EIGEN_USE_GPU
#include "paddle/operators/sigmoid_op.h"

REGISTER_OP_GPU_KERNEL(sigmoid, ops::SigmoidKernel<ops::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(sigmoid_grad, ops::SigmoidGradKernel<ops::GPUPlace, float>);
