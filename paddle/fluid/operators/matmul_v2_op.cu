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

#include "paddle/fluid/operators/matmul_v2_op.h"

namespace ops = paddle::operators;
namespace plf = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    matmul_v2, ops::MatMulV2Kernel<plf::CUDADeviceContext, float>,
    ops::MatMulV2Kernel<plf::CUDADeviceContext, double>,
    ops::MatMulV2Kernel<plf::CUDADeviceContext, plf::float16>,
    ops::MatMulV2Kernel<plf::CUDADeviceContext, plf::complex<float>>,
    ops::MatMulV2Kernel<plf::CUDADeviceContext, plf::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    matmul_v2_grad, ops::MatMulV2GradKernel<plf::CUDADeviceContext, float>,
    ops::MatMulV2GradKernel<plf::CUDADeviceContext, double>,
    ops::MatMulV2GradKernel<plf::CUDADeviceContext, plf::float16>,
    ops::MatMulV2GradKernel<plf::CUDADeviceContext, plf::complex<float>>,
    ops::MatMulV2GradKernel<plf::CUDADeviceContext, plf::complex<double>>);
