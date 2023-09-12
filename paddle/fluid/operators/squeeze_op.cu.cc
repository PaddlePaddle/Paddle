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

#include "paddle/fluid/operators/squeeze_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    squeeze,
    ops::SqueezeKernel<phi::GPUContext, float>,
    ops::SqueezeKernel<phi::GPUContext, double>,
    ops::SqueezeKernel<phi::GPUContext, plat::float16>,
    ops::SqueezeKernel<phi::GPUContext, plat::bfloat16>,
    ops::SqueezeKernel<phi::GPUContext, bool>,
    ops::SqueezeKernel<phi::GPUContext, int>,
    ops::SqueezeKernel<phi::GPUContext, uint8_t>,
    ops::SqueezeKernel<phi::GPUContext, int8_t>,
    ops::SqueezeKernel<phi::GPUContext, int64_t>,
    ops::SqueezeKernel<phi::GPUContext, paddle::platform::complex<float>>,
    ops::SqueezeKernel<phi::GPUContext, paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<phi::GPUContext, float>,
    ops::SqueezeGradKernel<phi::GPUContext, double>,
    ops::SqueezeGradKernel<phi::GPUContext, plat::float16>,
    ops::SqueezeGradKernel<phi::GPUContext, plat::bfloat16>,
    ops::SqueezeGradKernel<phi::GPUContext, bool>,
    ops::SqueezeGradKernel<phi::GPUContext, int>,
    ops::SqueezeGradKernel<phi::GPUContext, uint8_t>,
    ops::SqueezeGradKernel<phi::GPUContext, int8_t>,
    ops::SqueezeGradKernel<phi::GPUContext, int64_t>,
    ops::SqueezeGradKernel<phi::GPUContext, paddle::platform::complex<float>>,
    ops::SqueezeGradKernel<phi::GPUContext, paddle::platform::complex<double>>);
