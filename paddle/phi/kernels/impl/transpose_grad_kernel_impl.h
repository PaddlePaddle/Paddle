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

#pragma once

#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  size_t axis_size = axis.size();
  std::vector<int> formated_axis = axis;
  for (size_t i = 0; i < axis_size; i++) {
    if (axis[i] < 0) {
      formated_axis[i] = axis[i] + axis_size;
    }
  }

  std::vector<int> reversed_axis(axis);
  dev_ctx.template Alloc<T>(x_grad);
  for (size_t i = 0; i < axis_size; i++) {
    reversed_axis[formated_axis[i]] = i;
  }

  TransposeKernel<T, Context>(dev_ctx, out_grad, reversed_axis, x_grad);
}

template <typename T, typename Context>
void TransLayoutGradKernel(const Context& dev_ctx,
                           const DenseTensor& x UNUSED,
                           const DenseTensor& out_grad,
                           const std::vector<int>& axis,
                           DenseTensor* x_grad) {
  TransposeGradKernel<T, Context>(dev_ctx, out_grad, axis, x_grad);
}

}  // namespace phi
