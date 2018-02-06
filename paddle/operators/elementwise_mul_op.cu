/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include "paddle/operators/elementwise_mul_op.h"

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(
    elementwise_mul,
    ops::ElementwiseMulKernel<paddle::platform::GPUPlace, float>,
    ops::ElementwiseMulKernel<paddle::platform::GPUPlace, double>,
    ops::ElementwiseMulKernel<paddle::platform::GPUPlace, int>,
    ops::ElementwiseMulKernel<paddle::platform::GPUPlace, int64_t>);
REGISTER_OP_GPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<paddle::platform::GPUPlace, float>,
    ops::ElementwiseMulGradKernel<paddle::platform::GPUPlace, double>,
    ops::ElementwiseMulGradKernel<paddle::platform::GPUPlace, int>,
    ops::ElementwiseMulGradKernel<paddle::platform::GPUPlace, int64_t>);
