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

// .part used to speed up nvcc compile
#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"

template <typename T>
using CUDAReduceMeanGradKernel =
    ops::ReduceCudaGradKernel<T, kps::DivideFunctor>;

REGISTER_OP_CUDA_KERNEL(reduce_mean_grad, CUDAReduceMeanGradKernel<bool>,
                        CUDAReduceMeanGradKernel<paddle::platform::float16>,
                        CUDAReduceMeanGradKernel<float>,
                        CUDAReduceMeanGradKernel<double>);
