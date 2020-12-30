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

#include "paddle/fluid/operators/abs_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

/* ==========================   abs register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    abs, ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<float>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<double>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<int>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<int64_t>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<plat::float16>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<plat::complex64>>,
    ops::AbsKernel<plat::CUDADeviceContext, ops::AbsFunctor<plat::complex128>>);
REGISTER_OP_CUDA_KERNEL(
    abs_grad,
    ops::AbsGradKernel<plat::CUDADeviceContext, ops::AbsGradFunctor<float>>,
    ops::AbsGradKernel<plat::CUDADeviceContext, ops::AbsGradFunctor<double>>,
    ops::AbsGradKernel<plat::CUDADeviceContext, ops::AbsGradFunctor<int>>,
    ops::AbsGradKernel<plat::CUDADeviceContext, ops::AbsGradFunctor<int64_t>>,
    ops::AbsGradKernel<plat::CUDADeviceContext,
                       ops::AbsGradFunctor<plat::float16>>,
    ops::AbsGradKernel<plat::CUDADeviceContext,
                       ops::AbsGradFunctor<plat::complex64>>,
    ops::AbsGradKernel<plat::CUDADeviceContext,
                       ops::AbsGradFunctor<plat::complex128>>);
REGISTER_OP_CUDA_KERNEL(
    abs_grad_grad, ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                            ops::AbsGradGradFunctor<float>>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             ops::AbsGradGradFunctor<double>>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext,
                             ops::AbsGradGradFunctor<plat::float16>>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             ops::AbsGradGradFunctor<int>>,
    ops::AbsDoubleGradKernel<paddle::platform::CUDADeviceContext,
                             ops::AbsGradGradFunctor<int64_t>>);
/* ========================================================================== */
