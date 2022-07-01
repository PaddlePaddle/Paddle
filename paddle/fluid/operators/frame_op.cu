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

#include "paddle/fluid/operators/frame_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    frame,
    ops::FrameKernel<paddle::platform::CUDADeviceContext, int>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::float16>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::complex<float>>,
    ops::FrameKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    frame_grad,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::float16>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::FrameGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
