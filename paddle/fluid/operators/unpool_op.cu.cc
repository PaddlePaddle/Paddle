/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unpool_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    unpool, ops::UnpoolKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UnpoolKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    unpool_grad,
    ops::UnpoolGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UnpoolGradKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    unpool3d, ops::Unpool3dKernel<paddle::platform::CUDADeviceContext, float>,
    ops::Unpool3dKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    unpool3d_grad,
    ops::Unpool3dGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::Unpool3dGradKernel<paddle::platform::CUDADeviceContext, double>);
