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

#include "paddle/fluid/operators/stack_op.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    stack, ops::StackKernel<plat::CUDADeviceContext, float>,
    ops::StackKernel<plat::CUDADeviceContext, double>,
    ops::StackKernel<plat::CUDADeviceContext, int>,
    ops::StackKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackKernel<plat::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    stack_grad, ops::StackGradKernel<plat::CUDADeviceContext, float>,
    ops::StackGradKernel<plat::CUDADeviceContext, double>,
    ops::StackGradKernel<plat::CUDADeviceContext, int>,
    ops::StackGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackGradKernel<plat::CUDADeviceContext, plat::float16>);
