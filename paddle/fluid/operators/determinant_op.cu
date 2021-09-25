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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/determinant_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    determinant, ops::DeterminantKernel<plat::CUDADeviceContext, float>,
    ops::DeterminantKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    determinant_grad,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, float>,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    slogdeterminant, ops::SlogDeterminantKernel<plat::CUDADeviceContext, float>,
    ops::SlogDeterminantKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    slogdeterminant_grad,
    ops::SlogDeterminantGradKernel<plat::CUDADeviceContext, float>,
    ops::SlogDeterminantGradKernel<plat::CUDADeviceContext, double>);
