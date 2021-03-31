// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/abs_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    abs, ops::AbsKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext, double>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext, int>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext,
                   paddle::platform::float16>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext,
                   paddle::platform::complex64>,
    ops::AbsKernel<paddle::platform::CUDADeviceContext,
                   paddle::platform::complex128>);

REGISTER_OP_CUDA_KERNEL(
    abs_grad, ops::AbsGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext,
                       paddle::platform::float16>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext,
                       paddle::platform::complex64>,
    ops::AbsGradKernel<paddle::platform::CUDADeviceContext,
                       paddle::platform::complex128>);

REGISTER_OP_CUDA_KERNEL(
    abs_grad_grad,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::float16>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::complex64>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::complex128>);
