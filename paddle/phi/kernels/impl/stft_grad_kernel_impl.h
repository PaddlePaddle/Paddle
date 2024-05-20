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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"
#include "paddle/phi/kernels/funcs/frame_functor.h"
#include "paddle/phi/kernels/funcs/padding.h"

namespace phi {
// Multiply
template <typename T>
using MulFunctor = phi::funcs::MultiplyFunctor<T>;

// It is a common implementation to compute binary calculation with the support
// of broadcast, supporting both CPU and GPU.
// - CPU implementation cannot support the case when x needs broadcast, thus
//   this function need to be called with XxxFunctor and XxxInverseFunctor,
//   like AddFunctor and InverseAddFunctor.
// - GPU implementation supports all the broadcast cases, thus there is no need
//   to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor, typename Context, typename T, typename OutType = T>
void ElementwiseComputeEx(const Context& dev_ctx,
                          const phi::DenseTensor* x,
                          const phi::DenseTensor* y,
                          int axis,
                          Functor func,
                          phi::DenseTensor* z) {
  dev_ctx.template Alloc<OutType>(z);
  phi::funcs::ElementwiseCompute<Functor, T, OutType>(
      dev_ctx, *x, *y, func, z, axis);
}

template <typename T, typename Context>
void StftGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& window,
                    const DenseTensor& out_grad,
                    int n_fft,
                    int hop_length,
                    bool normalized,
                    bool onesided,
                    DenseTensor* x_grad) {
  using C = phi::dtype::complex<T>;

  const auto* dy = &out_grad;
  auto* dx = x_grad;
  dev_ctx.template Alloc<T>(x_grad);

  const size_t dy_rank = dy->dims().size();
  const size_t dx_rank = dx->dims().size();

  const int n_frames = dy->dims()[dy_rank - 1];
  const int seq_length = dx->dims()[dx_rank - 1];

  std::vector<int64_t> axes = {1};
  phi::DenseTensor d_frames_w;
  phi::DDim d_frames_dims(dy->dims());
  d_frames_dims.at(axes.back()) = n_fft;
  d_frames_w.Resize(d_frames_dims);
  dev_ctx.template Alloc<T>(&d_frames_w);

  phi::DenseTensor complex_d_frames_w;
  complex_d_frames_w.Resize(d_frames_dims);
  dev_ctx.template Alloc<C>(&complex_d_frames_w);

  // dy -> d_frames_w
  phi::funcs::FFTNormMode normalization;
  if (normalized) {
    normalization = phi::funcs::get_norm_from_string("ortho", true);
  } else {
    normalization = phi::funcs::get_norm_from_string("backward", true);
  }
  phi::funcs::FFTC2CFunctor<Context, C, C> fft_c2c_func;

  if (!onesided) {
    fft_c2c_func(dev_ctx, *dy, &complex_d_frames_w, axes, normalization, false);
  } else {
    phi::DenseTensor full_dy;
    full_dy.Resize(d_frames_dims);
    dev_ctx.template Alloc<C>(&full_dy);
    auto zero_length = static_cast<int>(full_dy.dims().at(axes.back()) -
                                        dy->dims().at(axes.back()));
    auto rank = dy->dims().size();

    std::vector<int> pads(rank * 2, 0);
    pads[axes.back() * 2 + 1] = zero_length;

    phi::funcs::PaddingFunctor<Context, C>(
        rank, dev_ctx, pads, static_cast<C>(0), *dy, &full_dy);
    fft_c2c_func(
        dev_ctx, full_dy, &complex_d_frames_w, axes, normalization, false);
  }
  phi::RealKernel<C>(dev_ctx, complex_d_frames_w, &d_frames_w);

  // d_frames_w -> d_frames
  phi::DenseTensor d_frames;
  d_frames.Resize(d_frames_dims);
  dev_ctx.template Alloc<T>(&d_frames);
  const phi::DenseTensor d_frames_w_const = d_frames_w;
  ElementwiseComputeEx<MulFunctor<T>, Context, T>(dev_ctx,
                                                  &d_frames_w_const,
                                                  &window,
                                                  axes.back(),
                                                  MulFunctor<T>(),
                                                  &d_frames);

  // d_frames -> dx
  phi::funcs::FrameFunctor<Context, T>()(dev_ctx,
                                         &d_frames,
                                         dx,
                                         seq_length,
                                         n_fft,
                                         n_frames,
                                         hop_length,
                                         /*is_grad*/ true);
}
}  // namespace phi
