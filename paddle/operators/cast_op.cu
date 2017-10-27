#include "paddle/operators/cast_op.h"

namespace ops = paddle::operators;
using GPU = paddle::platform::GPUPlace;

REGISTER_OP_GPU_KERNEL(cast, ops::CastOpKernel<GPU, float>,
                       ops::CastOpKernel<GPU, double>,
                       ops::CastOpKernel<GPU, int>,
                       ops::CastOpKernel<GPU, int64_t>);
