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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/elementwise_sub_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
