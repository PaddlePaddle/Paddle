#define EIGEN_USE_GPU

#include "paddle/operators/mean_op.h"

REGISTER_OP_GPU_KERNEL(mean, ops::MeanKernel<ops::GPUPlace, float>);
