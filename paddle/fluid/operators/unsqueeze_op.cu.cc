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
    unsqueeze, ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, plat::bfloat16>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    unsqueeze_grad,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext,
                             plat::bfloat16>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::complex<float>>,
    ops::UnsqueezeGradKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    unsqueeze2,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, plat::bfloat16>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::UnsqueezeKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    unsqueeze2_grad,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext,
                              plat::float16>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext,
                              plat::bfloat16>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<float>>,
    ops::Unsqueeze2GradKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<double>>);
