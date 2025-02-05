// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fake_dequantize_kernel.h"
#include "paddle/phi/kernels/funcs/fake_dequantize_functor.h"

namespace phi {

template <typename T, typename Context>
void FakeDequantizeMaxAbsKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& scale,
                                float max_range,
                                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  phi::funcs::DequantizeFunctor<Context, T>()(
      dev_ctx, &x, &scale, static_cast<T>(max_range), out);
}

template <typename T, typename Context>
void FakeChannelWiseDequantizeMaxAbsKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& scales,
    const std::vector<int>& quant_bits,
    int quant_axis,
    int x_num_col_dims,
    DenseTensor* out) {
  int max_range = 1;
  dev_ctx.template Alloc<T>(out);
  int scale_num = scales.size();
  if (scale_num == 1) {
    PADDLE_ENFORCE_EQ(
        scales[0]->numel(),
        x.dims()[quant_axis],
        common::errors::PreconditionNotMet(
            "The number of first scale values must be the same with "
            "quant_axis dimension value of Input(X) when the `Scales` has "
            "only one element, but %ld != %ld here.",
            scales[0]->numel(),
            x.dims()[quant_axis]));
    max_range *= (std::pow(2, quant_bits[0] - 1) - 1);
  } else if (scale_num == 2) {
    PADDLE_ENFORCE_EQ(
        scales[0]->numel(),
        x.dims()[x_num_col_dims],
        common::errors::PreconditionNotMet(
            "The number of first scale values must be the same with "
            "corresponding dimension value of Input(X) when the `Scales` "
            "has two elements, but %ld != %ld here.",
            scales[0]->numel(),
            x.dims()[1]));
    PADDLE_ENFORCE_EQ(scales[1]->numel(),
                      1,
                      common::errors::PreconditionNotMet(
                          "The second scale tensor should only have one "
                          "value at now, but it has %ld values here.",
                          scales[1]->numel()));
    max_range *= (std::pow(2, quant_bits[0] - 1) - 1) *
                 (std::pow(2, quant_bits[1] - 1) - 1);
  }
  phi::funcs::ChannelDequantizeFunctor<Context, T>()(
      dev_ctx,
      &x,
      (const_cast<std::vector<const DenseTensor*>*>(&scales))->data(),
      scale_num,
      static_cast<T>(max_range),
      quant_axis,
      x_num_col_dims,
      out);
}

}  // namespace phi
