// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "custom_raw_op_kernel_op.h"  // NOLINT
#include "paddle/fluid/framework/custom_raw_op_kernel_func.h"
#include "paddle/fluid/platform/enforce.h"

void ReluCPUForward(const phi::DenseTensor &x, phi::DenseTensor *y) {
  custom_raw_op::ReluForward(x, y);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void ReluGPUForward(const phi::DenseTensor &x, phi::DenseTensor *y);
#else
void ReluGPUForward(const phi::DenseTensor &x, phi::DenseTensor *y) {
  PADDLE_THROW(common::errors::Unimplemented(
      "ReluGPUForward is not supported when not compiled with GPU."));
}
#endif

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_raw_relu, ctx) {
  namespace f = paddle::framework;
  const auto *x = ctx.Input<phi::DenseTensor>("X");
  auto *y = ctx.Output<phi::DenseTensor>("Y");
  PADDLE_ENFORCE_NOT_NULL(
      x, common::errors::InvalidArgument("Input(X) should not be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      y, common::errors::InvalidArgument("Input(X) should not be nullptr."));
  if (phi::is_gpu_place(x->place())) {
    ReluGPUForward(*x, y);
  } else {
    ReluCPUForward(*x, y);
  }
}

PD_BUILD_OP(custom_raw_relu).Inputs({"X"}).Outputs({"Y"});
