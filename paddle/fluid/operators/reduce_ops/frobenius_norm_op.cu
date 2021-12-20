// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/frobenius_norm_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

template <typename T>
using CUDAFrobeniusNormKernel =
    ops::ReduceKernel<paddle::platform::CUDADeviceContext, T,
                      ops::FrobeniusNormFunctor>;

REGISTER_OP_CUDA_KERNEL(frobenius_norm, CUDAFrobeniusNormKernel<float>,
                        CUDAFrobeniusNormKernel<double>);

template <typename T>
using CUDAFrobeniusNormGradKernel =
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, T,
                          ops::FrobeniusNormGradFunctor>;

REGISTER_OP_CUDA_KERNEL(frobenius_norm_grad, CUDAFrobeniusNormGradKernel<float>,
                        CUDAFrobeniusNormGradKernel<double>);
