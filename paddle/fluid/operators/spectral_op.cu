/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/spectral_op.cu.h"
#include "paddle/fluid/operators/spectral_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2r, ops::FFTC2RKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2RKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2r_grad,
    ops::FFTC2RGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2RGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_r2c, ops::FFTR2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTR2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_r2c_grad,
    ops::FFTR2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTR2CGradKernel<paddle::platform::CUDADeviceContext, double>);
