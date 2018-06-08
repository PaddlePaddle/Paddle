/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/arg_max_op.h"
/*
REGISTER_ARG_MINMAX_OP_WITHOUT_GRADIENT(arg_max, ArgMax);

REGISTER_ARG_MINMAX_KERNEL(arg_max, ArgMax, CPU);
*/

REGISTER_OPERATOR(arg_max, paddle::operators::ArgMaxOp,
                  paddle::operators::ArgMaxOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(
    arg_max, paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext,
                                             float, int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, double,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, int64_t,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, int32_t,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, int16_t,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, size_t,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CPUDeviceContext, uint8_t,
                                    int64_t>);
