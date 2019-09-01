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

#include "paddle/fluid/operators/unstack_op.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    unstack, ops::UnStackKernel<plat::CUDADeviceContext, float>,
    ops::UnStackKernel<plat::CUDADeviceContext, double>,
    ops::UnStackKernel<plat::CUDADeviceContext, int>,
    ops::UnStackKernel<plat::CUDADeviceContext, int64_t>,
    ops::UnStackKernel<plat::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    unstack_grad, ops::UnStackGradKernel<plat::CUDADeviceContext, float>,
    ops::UnStackGradKernel<plat::CUDADeviceContext, double>,
    ops::UnStackGradKernel<plat::CUDADeviceContext, int>,
    ops::UnStackGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::UnStackGradKernel<plat::CUDADeviceContext, plat::float16>);
