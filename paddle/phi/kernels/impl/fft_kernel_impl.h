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
#include "paddle/phi/kernels/fft_kernel.h"

#include <string>
#include <vector>

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"

namespace phi {
template <typename T, typename Context>
void FFTC2CKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  DenseTensor* out) {
  ctx.template Alloc<T>(out);
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;
  fft_c2c_func(ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTC2RKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  int64_t last_dim_size UNUSED,
                  DenseTensor* out) {
  using R = typename T::value_type;  // get real type
  ctx.template Alloc<R>(out);
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2RFunctor<Context, T, R> fft_c2r_func;
  fft_c2r_func(ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTR2CKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  bool onesided,
                  DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  ctx.template Alloc<C>(out);
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;

  if (onesided) {
    fft_r2c_func(ctx, x, out, axes, norm_type, forward);
  } else {
    phi::DDim onesided_out_shape = x.dims();
    const int64_t last_fft_axis = axes.back();
    const int64_t onesided_last_axis_size =
        out->dims().at(last_fft_axis) / 2 + 1;
    onesided_out_shape[last_fft_axis] = onesided_last_axis_size;
    DenseTensor onesided_out =
        Empty<C, Context>(ctx, phi::vectorize(onesided_out_shape));
    fft_r2c_func(ctx, x, &onesided_out, axes, norm_type, forward);
    funcs::FFTFillConj<Context, C>(ctx, &onesided_out, out, axes);
  }
}
}  // namespace phi
