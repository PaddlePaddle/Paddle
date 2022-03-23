// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/spectral_op.cu.h"
#include "paddle/fluid/operators/stft_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    stft, ops::StftKernel<paddle::platform::CUDADeviceContext, float>,
    ops::StftKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    stft_grad, ops::StftGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::StftGradKernel<paddle::platform::CUDADeviceContext, double>);
