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
void UnfoldGradKernel(const Context& ctx,
                      const DenseTensor& x UNUSED,
                      const DenseTensor& out_grad,
                      const std::vector<int>& kernel_sizes,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  if (!x_grad) return;

  const auto& x_dims = x_grad->dims();
  const int batch_size = static_cast<int>(x_dims[0]);

  int out_height = phi::funcs::CalcOutputSize(x_dims[2],
                                              kernel_sizes[0],
                                              dilations[0],
                                              paddings[0],
                                              paddings[2],
                                              strides[0]);
  int out_width = phi::funcs::CalcOutputSize(x_dims[3],
                                             kernel_sizes[1],
                                             dilations[1],
                                             paddings[1],
                                             paddings[3],
                                             strides[1]);

  DDim x_shape = make_ddim({x_dims[1], x_dims[2], x_dims[3]});
  DDim out_matrix_shape = make_ddim(
      {x_dims[1], kernel_sizes[0], kernel_sizes[1], out_height, out_width});

  phi::funcs::Col2ImFunctor<phi::funcs::ColFormat::kCFO, Context, T> col2im;

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(ctx, x_grad, static_cast<T>(0));
  for (int i = 0; i < batch_size; i++) {
    DenseTensor out_grad_batch =
        out_grad.Slice(i, i + 1).Resize(out_matrix_shape);
    DenseTensor x_grad_batch = x_grad->Slice(i, i + 1).Resize(x_shape);
    col2im(ctx, out_grad_batch, dilations, strides, paddings, &x_grad_batch);
  }
}

}  // namespace phi
