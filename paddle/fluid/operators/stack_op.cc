// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/stack_op.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;
REGISTER_OPERATOR(stack, ops::StackOp, ops::StackOpMaker,
                  ops::StackGradOpDescMaker);
REGISTER_OPERATOR(stack_grad, ops::StackOpGrad);

REGISTER_OP_CPU_KERNEL(stack, ops::StackKernel<plat::CPUDeviceContext, float>,
                       ops::StackKernel<plat::CPUDeviceContext, double>,
                       ops::StackKernel<plat::CPUDeviceContext, int>,
                       ops::StackKernel<plat::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(stack_grad,
                       ops::StackGradKernel<plat::CPUDeviceContext, float>,
                       ops::StackGradKernel<plat::CPUDeviceContext, double>,
                       ops::StackGradKernel<plat::CPUDeviceContext, int>,
                       ops::StackGradKernel<plat::CPUDeviceContext, int64_t>);
