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

#include "paddle/fluid/operators/eigvalsh_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    eigvalsh,
    ops::EigvalshKernel<paddle::platform::CUDADeviceContext, float, float>,
    ops::EigvalshKernel<paddle::platform::CUDADeviceContext, double, double>,
    ops::EigvalshKernel<paddle::platform::CUDADeviceContext,
                        float,
                        paddle::platform::complex<float>>,
    ops::EigvalshKernel<paddle::platform::CUDADeviceContext,
                        double,
                        paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    eigvalsh_grad,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, float, float>,
    ops::
        EigvalshGradKernel<paddle::platform::CUDADeviceContext, double, double>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext,
                            float,
                            paddle::platform::complex<float>>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext,
                            double,
                            paddle::platform::complex<double>>);
