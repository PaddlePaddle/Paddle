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

#include "paddle/fluid/operators/pool_with_index_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    max_pool2d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CUDADeviceContext, float,
                                int>,
    ops::MaxPoolWithIndexKernel<paddle::platform::CUDADeviceContext, double,
                                int>);
REGISTER_OP_CUDA_KERNEL(
    max_pool2d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CUDADeviceContext, float,
                                    int>,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CUDADeviceContext, double,
                                    int>);

REGISTER_OP_CUDA_KERNEL(
    max_pool3d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CUDADeviceContext, float,
                                int>,
    ops::MaxPoolWithIndexKernel<paddle::platform::CUDADeviceContext, double,
                                int>);
REGISTER_OP_CUDA_KERNEL(
    max_pool3d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CUDADeviceContext, float,
                                    int>,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CUDADeviceContext, double,
                                    int>);
