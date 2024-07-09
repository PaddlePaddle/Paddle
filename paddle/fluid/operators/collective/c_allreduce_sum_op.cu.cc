/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle {
namespace operators {
DEFINE_C_ALLREDUCE_CUDA_KERNEL(CAllReduceSum, kRedSum)
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(c_allreduce_sum,
                          GPU,
                          ALL_LAYOUT,
                          ops::CAllReduceSumCUDAKernel,
                          float,
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
                          phi::dtype::bfloat16,
#endif
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {
}
