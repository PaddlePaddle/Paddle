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

#include "paddle/fluid/operators/eigh_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    eigh, ops::EighKernel<paddle::platform::CUDADeviceContext, float>,
    ops::EighKernel<paddle::platform::CUDADeviceContext, double>,
    ops::EighKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::complex<float>>,
    ops::EighKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    eigh_grad, ops::EighGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::complex<float>>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::complex<double>>);
