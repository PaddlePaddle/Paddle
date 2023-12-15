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

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

namespace phi {

template <typename T, typename Context>
void FoldGradKernel(const Context& ctx,
                    const DenseTensor& x UNUSED,
                    const DenseTensor& out_grad,
                    const std::vector<int>& output_sizes,
                    const std::vector<int>& kernel_sizes,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    const std::vector<int>& dilations,
                    DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  if (!x_grad) return;

  const auto& x_dims = x_grad->dims();
  const int batch_size = static_cast<int>(x_dims[0]);

  int output_height = (output_sizes[0] + 2 * paddings[0] -
                       (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                          strides[0] +
                      1;
  int output_width = (output_sizes[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                         strides[1] +
                     1;

  int n_input_plane = x_dims[1];
  int n_output_plane = n_input_plane / (kernel_sizes[0] * kernel_sizes[1]);

  DDim out_shape =
      common::make_ddim({n_output_plane, output_sizes[0], output_sizes[1]});
  DDim input_matrix_shape = common::make_ddim(
      {1, kernel_sizes[0], kernel_sizes[1], output_height, output_width});

  phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kCFO, Context, T> im2col;

  for (int i = 0; i < batch_size; i++) {
    DenseTensor out_grad_batch = out_grad.Slice(i, i + 1).Resize(out_shape);
    DenseTensor x_grad_batch =
        x_grad->Slice(i, i + 1).Resize(input_matrix_shape);
    im2col(ctx, out_grad_batch, dilations, strides, paddings, &x_grad_batch);
  }
}

}  // namespace phi
