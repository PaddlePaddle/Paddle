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

REGISTER_OP_CPU_KERNEL(
    abs, paddle::framework::CUDAKernelCont<
             paddle::operators::AbsKernel, float, double, int, int64_t,
             paddle::platform::float16, paddle::platform::complex64,
             paddle::platform::complex128>);

REGISTER_OP_CPU_KERNEL(
    abs_grad, paddle::framework::CUDAKernelCont<
                  paddle::operators::AbsGradKernel, float, double, int, int64_t,
                  paddle::platform::float16, paddle::platform::complex64,
                  paddle::platform::complex128>);

REGISTER_OP_CPU_KERNEL(
    abs_grad_grad,
    paddle::framework::CUDAKernelCont<
        paddle::operators::AbsDoubleGradKernel, float, double, int, int64_t,
        paddle::platform::float16, paddle::platform::complex64,
        paddle::platform::complex128>);
