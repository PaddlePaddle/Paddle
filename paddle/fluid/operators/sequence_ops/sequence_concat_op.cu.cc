// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/sequence_ops/sequence_concat_op.h"

#include "paddle/fluid/framework/op_registry.h"

REGISTER_OP_CUDA_KERNEL(
    sequence_concat,
    paddle::operators::SeqConcatKernel<phi::GPUContext, float>,
    paddle::operators::SeqConcatKernel<phi::GPUContext, double>,
    paddle::operators::SeqConcatKernel<phi::GPUContext, int>,
    paddle::operators::SeqConcatKernel<phi::GPUContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    sequence_concat_grad,
    paddle::operators::SeqConcatGradKernel<phi::GPUContext, float>,
    paddle::operators::SeqConcatGradKernel<phi::GPUContext, double>,
    paddle::operators::SeqConcatGradKernel<phi::GPUContext, int>,
    paddle::operators::SeqConcatGradKernel<phi::GPUContext, int64_t>);
