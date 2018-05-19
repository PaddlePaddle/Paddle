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

#include "paddle/fluid/operators/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise_op.h"
namespace ops = paddle::operators;
REGISTER_ELEMWISE_OP(elementwise_add, "Add", "Out = X + Y");
REGISTER_OP_CPU_KERNEL(
    elementwise_add,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
