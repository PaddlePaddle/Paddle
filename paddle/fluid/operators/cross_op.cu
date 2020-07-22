// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/cross_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cross, ops::CrossKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CrossKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CrossKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CrossKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    cross_grad,
    ops::CrossGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CrossGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CrossGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CrossGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
