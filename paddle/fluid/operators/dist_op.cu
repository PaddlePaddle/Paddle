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

#include "paddle/fluid/operators/dist_op.h"

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_HIP
// Eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:922
// do not support double in HIPCC platform (Eigen3 to be fixed)
REGISTER_OP_CUDA_KERNEL(
    dist, ops::DistKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(
    dist, ops::DistKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DistKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DistGradKernel<paddle::platform::CUDADeviceContext, double>);
#endif
