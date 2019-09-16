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

REGISTER_OP_CUDA_KERNEL(
    squeeze, ops::SqueezeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SqueezeGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SqueezeGradKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::SqueezeGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    squeeze2, ops::Squeeze2Kernel<paddle::platform::CUDADeviceContext, float>,
    ops::Squeeze2Kernel<paddle::platform::CUDADeviceContext, double>,
    ops::Squeeze2Kernel<paddle::platform::CUDADeviceContext, int>,
    ops::Squeeze2Kernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::Squeeze2Kernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::Squeeze2GradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::Squeeze2GradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::Squeeze2GradKernel<paddle::platform::CUDADeviceContext, int8_t>,
    ops::Squeeze2GradKernel<paddle::platform::CUDADeviceContext, int64_t>);
