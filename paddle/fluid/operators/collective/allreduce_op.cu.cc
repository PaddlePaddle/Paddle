/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/allreduce_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(allreduce,
                        ops::AllReduceOpKernel<phi::GPUContext, float>,
                        ops::AllReduceOpKernel<phi::GPUContext, double>,
                        ops::AllReduceOpKernel<phi::GPUContext, int>,
                        ops::AllReduceOpKernel<phi::GPUContext, int64_t>,
                        ops::AllReduceOpKernel<phi::GPUContext, plat::float16>);
