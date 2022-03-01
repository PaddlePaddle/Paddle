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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

using CUDA = paddle::platform::CUDADeviceContext;
#define REGISTER_CAST_CUDA_BASE(op_name, ...)                             \
  REGISTER_OP_CUDA_KERNEL(                                                \
      op_name, ops::CastOpKernel<CUDA, float>,                            \
      ops::CastOpKernel<CUDA, double>, ops::CastOpKernel<CUDA, int>,      \
      ops::CastOpKernel<CUDA, int64_t>, ops::CastOpKernel<CUDA, int16_t>, \
      ops::CastOpKernel<CUDA, bool>, ops::CastOpKernel<CUDA, uint8_t>,    \
      ops::CastOpKernel<CUDA, plat::float16>,                             \
      ops::CastOpKernel<CUDA, plat::complex<float>>,                      \
      ops::CastOpKernel<CUDA, plat::complex<double>>, ##__VA_ARGS__);

// See [ why register transfer_dtype_op alias with cast_op? ] in cast_op.cc
REGISTER_CAST_CUDA_BASE(transfer_dtype, ops::CastOpKernel<CUDA, plat::bfloat16>)
