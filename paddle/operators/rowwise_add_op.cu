#include "paddle/operators/rowwise_add_op.h"

REGISTER_OP_GPU_KERNEL(rowwise_add,
                       ops::RowwiseAddKernel<ops::GPUPlace, float>);
