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

#include "paddle/fluid/operators/reshape_op.h"
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL_EX(reshape, float, ops::ReshapeKernel, double,
                           ops::ReshapeKernel, int, ops::ReshapeKernel, int64_t,
                           ops::ReshapeKernel);
REGISTER_OP_CUDA_KERNEL(reshape_grad,
                        paddle::operators::ReshapeGradKernel<float>,
                        paddle::operators::ReshapeGradKernel<double>,
                        paddle::operators::ReshapeGradKernel<int>,
                        paddle::operators::ReshapeGradKernel<int64_t>);
