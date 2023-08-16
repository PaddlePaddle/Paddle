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

#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/tril_triu_compute.h"
#include "paddle/phi/kernels/tril_triu_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void TrilTriuGradKernel(const Context& ctx,
                        const DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        DenseTensor* x_grad) {
  const auto* dout_data = out_grad.data<T>();
  auto* dx_data = ctx.template Alloc<T>(x_grad);

  const auto& dims = out_grad.dims();
  const auto H = dims[dims.size() - 2];
  const auto W = dims[dims.size() - 1];

  phi::funcs::ForRange<Context> for_range(
      ctx, static_cast<size_t>(out_grad.numel()));
  phi::funcs::TrilTriuCompute<T> tril_triu_grad_computer(
      dout_data, diagonal, lower, H, W, dx_data);
  for_range(tril_triu_grad_computer);
}

template <typename T, typename Context>
void TrilGradKernel(const Context& ctx,
                    const DenseTensor& out_grad,
                    int diagonal,
                    DenseTensor* x_grad) {
  TrilTriuGradKernel<T, Context>(ctx, out_grad, diagonal, true, x_grad);
}

template <typename T, typename Context>
void TriuGradKernel(const Context& ctx,
                    const DenseTensor& out_grad,
                    int diagonal,
                    DenseTensor* x_grad) {
  TrilTriuGradKernel<T, Context>(ctx, out_grad, diagonal, false, x_grad);
}

}  // namespace phi
