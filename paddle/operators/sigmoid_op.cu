#include "paddle/operators/sigmoid_op.h"

REGISTER_OP_GPU_KERNEL(sigmoid, ops::SigmoidKernel<ops::GPUPlace, float>);
