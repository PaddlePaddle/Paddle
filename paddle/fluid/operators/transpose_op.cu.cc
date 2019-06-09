/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/transpose_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    transpose, ops::TransposeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    transpose_grad,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    transpose2,
    ops::TransposeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    transpose2_grad,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);
