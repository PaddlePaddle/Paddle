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
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    isinf, ops::OverflowKernel<paddle::platform::CUDADeviceContext, int,
                               ops::InfinityFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, float,
                        ops::InfinityFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, double,
                        ops::InfinityFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, plat::float16,
                        ops::InfinityFunctor>);

REGISTER_OP_CUDA_KERNEL(isnan,
                        ops::OverflowKernel<paddle::platform::CUDADeviceContext,
                                            int, ops::NANFunctor>,
                        ops::OverflowKernel<paddle::platform::CUDADeviceContext,
                                            float, ops::NANFunctor>,
                        ops::OverflowKernel<paddle::platform::CUDADeviceContext,
                                            double, ops::NANFunctor>,
                        ops::OverflowKernel<paddle::platform::CUDADeviceContext,
                                            plat::float16, ops::NANFunctor>);

REGISTER_OP_CUDA_KERNEL(
    isfinite, ops::OverflowKernel<paddle::platform::CUDADeviceContext, int,
                                  ops::IsfiniteFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, float,
                        ops::IsfiniteFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, double,
                        ops::IsfiniteFunctor>,
    ops::OverflowKernel<paddle::platform::CUDADeviceContext, plat::float16,
                        ops::IsfiniteFunctor>);
