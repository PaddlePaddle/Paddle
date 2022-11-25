/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fill_any_op.h"
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    fill_any,
    ops::FillAnyKernel<phi::GPUContext, float>,
    ops::FillAnyKernel<phi::GPUContext, double>,
    ops::FillAnyKernel<phi::GPUContext, int64_t>,
    ops::FillAnyKernel<phi::GPUContext, int>,
    ops::FillAnyKernel<phi::GPUContext, paddle::platform::float16>,
    ops::FillAnyKernel<phi::GPUContext, bool>);

REGISTER_OP_CUDA_KERNEL(
    fill_any_grad,
    ops::FillAnyGradKernel<phi::GPUContext, float>,
    ops::FillAnyGradKernel<phi::GPUContext, double>,
    ops::FillAnyGradKernel<phi::GPUContext, int64_t>,
    ops::FillAnyGradKernel<phi::GPUContext, int>,
    ops::FillAnyGradKernel<phi::GPUContext, paddle::platform::float16>,
    ops::FillAnyGradKernel<phi::GPUContext, bool>);
