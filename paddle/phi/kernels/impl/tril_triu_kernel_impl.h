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
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

template <typename T, typename Context>
void TrilTriuKernel(const Context& ctx,
                    const DenseTensor& x,
                    int diagonal,
                    bool lower,
                    DenseTensor* out) {
  const auto* x_data = x.data<T>();
  auto* out_data = ctx.template Alloc<T>(out);

  const auto& dims = x.dims();
  const auto H = dims[dims.size() - 2];
  const auto W = dims[dims.size() - 1];
  phi::funcs::ForRange<Context> for_range(ctx, static_cast<size_t>(x.numel()));

  phi::funcs::TrilTriuCompute<T> tril_triu_computer(
      x_data, diagonal, lower, H, W, out_data);
  for_range(tril_triu_computer);
}

template <typename T, typename Context>
void TrilKernel(const Context& ctx,
                const DenseTensor& x,
                int diagonal,
                DenseTensor* out) {
  TrilTriuKernel<T, Context>(ctx, x, diagonal, true, out);
}

template <typename T, typename Context>
void TriuKernel(const Context& ctx,
                const DenseTensor& x,
                int diagonal,
                DenseTensor* out) {
  TrilTriuKernel<T, Context>(ctx, x, diagonal, false, out);
}

}  // namespace phi
