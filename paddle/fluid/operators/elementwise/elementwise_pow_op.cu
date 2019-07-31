/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/elementwise/elementwise_pow_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    elementwise_pow,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_pow_grad,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
