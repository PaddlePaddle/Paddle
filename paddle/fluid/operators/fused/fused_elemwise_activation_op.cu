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

#include "paddle/fluid/operators/fused/fused_elemwise_activation_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_elemwise_activation,
    ops::FusedElemwiseActivationKernel<paddle::platform::CUDADeviceContext,
                                       float>,
    ops::FusedElemwiseActivationKernel<paddle::platform::CUDADeviceContext,
                                       double>,
    ops::FusedElemwiseActivationKernel<paddle::platform::CUDADeviceContext,
                                       paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(
    fused_elemwise_activation_grad,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CUDADeviceContext,
                                           float>,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CUDADeviceContext,
                                           double>,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CUDADeviceContext,
                                           paddle::platform::float16>);
