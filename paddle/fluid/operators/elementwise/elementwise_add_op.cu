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

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int64_t>);
