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

#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
namespace ops = paddle::operators;
REGISTER_ELEMWISE_GRAD_MAKER(elementwise_sub, Sub);
REGISTER_ELEMWISE_EXPLICIT_OP(elementwise_sub, "Sub", "Out = X - Y");

REGISTER_OP_CPU_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
