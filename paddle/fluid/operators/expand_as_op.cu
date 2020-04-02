/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/expand_as_op.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    expand_as, ops::ExpandAsKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ExpandAsKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ExpandAsKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ExpandAsKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ExpandAsKernel<paddle::platform::CUDADeviceContext, bool>);
REGISTER_OP_CUDA_KERNEL(
    expand_as_grad,
    ops::ExpandAsGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ExpandAsGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ExpandAsGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ExpandAsGradKernel<paddle::platform::CUDADeviceContext, double>);
