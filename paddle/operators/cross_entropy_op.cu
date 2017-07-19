#include "paddle/operators/cross_entropy_op.h"
#include "paddle/framework/op_registry.h"

typedef paddle::operators::CrossEntropyOpKernel<::paddle::platform::GPUPlace, float>
    CrossEntropyOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(cross_entropy,
                       CrossEntropyOpKernel_GPU_float);