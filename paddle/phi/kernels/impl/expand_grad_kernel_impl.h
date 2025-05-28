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

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/expand_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ExpandGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& shape,
                      DenseTensor* in_grad) {
  if (x.numel() == 0 || out_grad.numel() == 0 ||
      (in_grad && in_grad->numel() == 0)) {
    ctx.template Alloc<T>(in_grad);
    if (in_grad->numel() != 0) {
      phi::Full<T, Context>(
          ctx, phi::IntArray(common::vectorize(in_grad->dims())), 0, in_grad);
    }
    return;
  }

  if (in_grad->dims() == out_grad.dims()) {
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, in_grad);
    return;
  }
  std::vector<int> reduce_dims =
      funcs::GetReduceDim(in_grad->dims(), out_grad.dims(), -1);
  phi::SumKernel<T, Context>(
      ctx, out_grad, reduce_dims, out_grad.dtype(), false, in_grad);
}

}  // namespace phi
