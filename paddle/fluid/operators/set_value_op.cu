//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/set_value_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    set_value, ops::SetValueKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SetValueKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SetValueKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SetValueKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SetValueKernel<paddle::platform::CUDADeviceContext, bool>);

REGISTER_OP_CUDA_KERNEL(
    set_value_grad,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, bool>);
