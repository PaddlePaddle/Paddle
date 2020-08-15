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
#include "paddle/fluid/operators/masked_select_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    masked_select,
    ops::MaskedSelectKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskedSelectKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    masked_select_grad,
    ops::MaskedSelectGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskedSelectGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
