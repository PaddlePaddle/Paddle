/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gelu, ops::GeluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    gelu_grad, ops::GeluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
