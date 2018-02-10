/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/conv_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d,
    ops::DepthwiseConvKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DepthwiseConvKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad,
    ops::DepthwiseConvGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DepthwiseConvGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    conv2d, ops::GemmConvKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    conv3d, ops::GemmConvKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    conv3d_grad,
    ops::GemmConvGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CUDADeviceContext, double>);
