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

#include "paddle/fluid/operators/reduce_sum_op.h"

REGISTER_REDUCE_OP(reduce_sum);
REGISTER_OP_CPU_KERNEL(
    reduce_sum, ops::ReduceKernel<paddle::platform::CPUDeviceContext, float,
                                  ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, double,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int, ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int64_t,
                      ops::SumFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_sum_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             float, ops::SumGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             double, ops::SumGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             int, ops::SumGradFunctor>,
                       ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                             int64_t, ops::SumGradFunctor>);
