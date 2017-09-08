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
#include "paddle/operators/reduce_op.h"

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(
    reduce_sum,
    ops::ReduceKernel<paddle::platform::GPUPlace, float, ops::SumFunctor>);
REGISTER_OP_GPU_KERNEL(reduce_sum_grad,
                       ops::ReduceGradEigenKernel<paddle::platform::GPUPlace,
                                                  float, ops::SumGradFunctor>);

REGISTER_OP_GPU_KERNEL(
    reduce_mean,
    ops::ReduceKernel<paddle::platform::GPUPlace, float, ops::MeanFunctor>);
REGISTER_OP_GPU_KERNEL(reduce_mean_grad,
                       ops::ReduceGradKernel<paddle::platform::GPUPlace, float,
                                             ops::MeanGradFunctor>);

REGISTER_OP_GPU_KERNEL(
    reduce_max,
    ops::ReduceKernel<paddle::platform::GPUPlace, float, ops::MaxFunctor>);
REGISTER_OP_GPU_KERNEL(reduce_max_grad,
                       ops::ReduceGradKernel<paddle::platform::GPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);

REGISTER_OP_GPU_KERNEL(
    reduce_min,
    ops::ReduceKernel<paddle::platform::GPUPlace, float, ops::MinFunctor>);
REGISTER_OP_GPU_KERNEL(reduce_min_grad,
                       ops::ReduceGradKernel<paddle::platform::GPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);