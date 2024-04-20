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

#include "paddle/phi/kernels/fake_quantize_kernel.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi {

template <typename T, typename Context>
void FakeQuantizeRangeAbsMaxKernel(const Context &dev_ctx,
                                   const DenseTensor &x,
                                   const DenseTensor &in_scale,
                                   const paddle::optional<DenseTensor> &iter,
                                   int window_size,
                                   int bit_length,
                                   bool is_test,
                                   int round_type,
                                   DenseTensor *out,
                                   DenseTensor *out_scale,
                                   DenseTensor *out_scales) {
  dev_ctx.template Alloc<T>(out);
  int bin_cnt = std::pow(2, bit_length - 1) - 1;

  // testing
  if (is_test) {
    phi::funcs::ClipAndFakeQuantFunctor<Context, T>()(
        dev_ctx, x, in_scale, bin_cnt, round_type, out);
    return;
  }

  // training
  dev_ctx.template Alloc<T>(out_scale);

  DenseTensor cur_scale;
  cur_scale.Resize({1});
  T *cur_scale_data = dev_ctx.template Alloc<T>(&cur_scale);
  phi::funcs::FindAbsMaxFunctor<Context, T>()(
      dev_ctx, x.data<T>(), x.numel(), cur_scale_data);
  phi::funcs::FindRangeAbsMaxFunctor<Context, T>()(dev_ctx,
                                                   cur_scale,
                                                   in_scale,
                                                   iter.get(),
                                                   window_size,
                                                   out_scales,
                                                   out_scale);
  phi::funcs::ClipAndFakeQuantFunctor<Context, T>()(
      dev_ctx, x, *out_scale, bin_cnt, round_type, out);
}

template <typename T, typename Context>
void FakeQuantizeAbsMaxKernel(const Context &dev_ctx,
                              const DenseTensor &x,
                              int bit_length,
                              int round_type,
                              DenseTensor *out,
                              DenseTensor *out_scale) {
  T *out_s = dev_ctx.template Alloc<T>(out_scale);
  int bin_cnt = std::pow(2, bit_length - 1) - 1;
  const T *in_data = x.data<T>();
  phi::funcs::FindAbsMaxFunctor<Context, T> find_abs_max_functor;
  find_abs_max_functor(dev_ctx, in_data, x.numel(), out_s);

  phi::funcs::ClipAndFakeQuantFunctor<Context, T> clip_and_fake_quant_functor;
  clip_and_fake_quant_functor(dev_ctx, x, *out_scale, bin_cnt, round_type, out);
}

}  // namespace phi
