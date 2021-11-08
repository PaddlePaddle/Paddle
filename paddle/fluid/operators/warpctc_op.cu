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

#include "paddle/fluid/operators/warpctc_op.h"

namespace ops = paddle::operators;
// register forward and backward of CUDA OP must in same *.cu file.
// Eigen can be used on GPU device, but must be in *.cu file not *.cu.cc file.
// *.cu.cc also using GCC compiler. *.cu using NVCC compiler
REGISTER_OP_CUDA_KERNEL(
    warpctc, ops::WarpCTCKernel<paddle::platform::CUDADeviceContext, float>,
    ops::WarpCTCKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    warpctc_grad,
    ops::WarpCTCGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::WarpCTCGradKernel<paddle::platform::CUDADeviceContext, double>);