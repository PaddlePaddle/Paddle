// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"
#include "paddle/phi/kernels/funcs/frame_functor.h"

namespace phi {

template <typename T, typename Context>
void StftKernel(const Context& ctx,
                const DenseTensor& x,
                const DenseTensor& window,
                int n_fft,
                int hop_length,
                bool normalized,
                bool onesided,
                DenseTensor* out) {
  using C = phi::dtype::complex<T>;

  ctx.template Alloc<C>(out);

  const size_t x_rank = x.dims().size();
  const size_t out_rank = out->dims().size();

  const int n_frames = out->dims()[out_rank - 1];
  const int seq_length = x.dims()[x_rank - 1];

  std::vector<int64_t> axes = {1};

  // Frame
  phi::DenseTensor frames;
  phi::DDim frames_dims(out->dims());
  frames_dims.at(axes.back()) = n_fft;
  frames.Resize(frames_dims);
  ctx.template Alloc<T>(&frames);
  phi::funcs::FrameFunctor<Context, T>()(ctx,
                                         &x,
                                         &frames,
                                         seq_length,
                                         n_fft,
                                         n_frames,
                                         hop_length,
                                         /*is_grad*/ false);

  // Window
  phi::DenseTensor frames_w;
  frames_w.Resize(frames_dims);
  ctx.template Alloc<T>(&frames_w);
  phi::funcs::ElementwiseCompute<phi::funcs::MultiplyFunctor<T>, T, T>(
      ctx,
      frames,
      window,
      phi::funcs::MultiplyFunctor<T>(),
      &frames_w,
      axes.back());

  // FFTR2C
  phi::funcs::FFTNormMode normalization;
  if (normalized) {
    normalization = phi::funcs::get_norm_from_string("ortho", true);
  } else {
    normalization = phi::funcs::get_norm_from_string("backward", true);
  }
  phi::funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;

  if (onesided) {
    fft_r2c_func(ctx, frames_w, out, axes, normalization, true);
  } else {
    phi::DDim onesided_dims(out->dims());
    const int64_t onesided_axis_size = out->dims().at(axes.back()) / 2 + 1;
    onesided_dims.at(axes.back()) = onesided_axis_size;
    phi::DenseTensor onesided_out;
    onesided_out.Resize(onesided_dims);
    ctx.template Alloc<T>(&onesided_out);
    fft_r2c_func(ctx, frames_w, &onesided_out, axes, normalization, true);
    phi::funcs::FFTFillConj<Context, C>(ctx, &onesided_out, out, axes);
  }
}
}  // namespace phi
