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

#include "paddle/fluid/operators/unsqueeze_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    unsqueeze,
    ops::UnsqueezeKernel<phi::GPUContext, float>,
    ops::UnsqueezeKernel<phi::GPUContext, double>,
    ops::UnsqueezeKernel<phi::GPUContext, plat::float16>,
    ops::UnsqueezeKernel<phi::GPUContext, plat::bfloat16>,
    ops::UnsqueezeKernel<phi::GPUContext, bool>,
    ops::UnsqueezeKernel<phi::GPUContext, int>,
    ops::UnsqueezeKernel<phi::GPUContext, int16_t>,
    ops::UnsqueezeKernel<phi::GPUContext, uint8_t>,
    ops::UnsqueezeKernel<phi::GPUContext, int8_t>,
    ops::UnsqueezeKernel<phi::GPUContext, int64_t>,
    ops::UnsqueezeKernel<phi::GPUContext, paddle::platform::complex<float>>,
    ops::UnsqueezeKernel<phi::GPUContext, paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    unsqueeze_grad,
    ops::UnsqueezeGradKernel<phi::GPUContext, float>,
    ops::UnsqueezeGradKernel<phi::GPUContext, double>,
    ops::UnsqueezeGradKernel<phi::GPUContext, plat::float16>,
    ops::UnsqueezeGradKernel<phi::GPUContext, plat::bfloat16>,
    ops::UnsqueezeGradKernel<phi::GPUContext, bool>,
    ops::UnsqueezeGradKernel<phi::GPUContext, int>,
    ops::UnsqueezeGradKernel<phi::GPUContext, int16_t>,
    ops::UnsqueezeGradKernel<phi::GPUContext, int8_t>,
    ops::UnsqueezeGradKernel<phi::GPUContext, uint8_t>,
    ops::UnsqueezeGradKernel<phi::GPUContext, int64_t>,
    ops::UnsqueezeGradKernel<phi::GPUContext, paddle::platform::complex<float>>,
    ops::UnsqueezeGradKernel<phi::GPUContext,
                             paddle::platform::complex<double>>);
