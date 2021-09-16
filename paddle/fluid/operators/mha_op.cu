/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mha_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mha,
    ops::MHAKernel<plat::CUDADeviceContext, plat::float16>,
    ops::MHAKernel<plat::CUDADeviceContext, float>,
    ops::MHAKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    mha_grad,
    ops::MHAGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::MHAGradKernel<plat::CUDADeviceContext, float>,
    ops::MHAGradKernel<plat::CUDADeviceContext, double>);
