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

#include "paddle/fluid/operators/fill_any_op.h"
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    fill_any,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext, int>,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext,
                       paddle::platform::float16>,
    ops::FillAnyKernel<paddle::platform::CUDADeviceContext, bool>);

REGISTER_OP_CUDA_KERNEL(
    fill_any_grad,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext,
                           paddle::platform::float16>,
    ops::FillAnyGradKernel<paddle::platform::CUDADeviceContext, bool>);
