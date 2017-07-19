#include "paddle/operators/add_op.h"

typedef paddle::operators::AddKernel<::paddle::platform::GPUPlace, float> AddKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(add_two,
                       AddKernel_GPU_float);