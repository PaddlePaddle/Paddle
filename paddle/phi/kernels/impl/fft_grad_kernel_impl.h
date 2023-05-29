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
#include "paddle/phi/kernels/fft_grad_kernel.h"

#include <string>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/pad_kernel.h"

namespace phi {
template <typename T, typename Context>
void FFTC2CGradKernel(const Context& ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;
  fft_c2c_func(ctx, out_grad, x_grad, axes, norm_type, !forward);
}

template <typename T, typename Context>
void FFTR2CGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      bool onesided,
                      DenseTensor* x_grad) {
  using R = typename T::value_type;
  DenseTensor complex_x_grad = EmptyLike<T>(ctx, x);
  ctx.template Alloc<R>(x_grad);
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;

  if (!onesided) {
    fft_c2c_func(ctx, out_grad, &complex_x_grad, axes, norm_type, !forward);
  } else {
    DenseTensor full_dy;
    DenseTensorMeta full_dy_meta(out_grad.type(), x_grad->dims());
    full_dy.set_meta(full_dy_meta);
    auto zero_length = static_cast<int>(full_dy.dims().at(axes.back()) -
                                        out_grad.dims().at(axes.back()));
    auto rank = out_grad.dims().size();
    std::vector<int> pads(rank * 2, 0);
    pads[axes.back() * 2 + 1] = zero_length;
    PadKernel<T>(ctx, out_grad, pads, static_cast<float>(0.0), &full_dy);
    fft_c2c_func(ctx, full_dy, &complex_x_grad, axes, norm_type, !forward);
  }
  RealKernel<T>(ctx, complex_x_grad, x_grad);
}

template <typename T, typename Context>
void FFTC2RGradKernel(const Context& ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      int64_t last_dim_size UNUSED,
                      DenseTensor* x_grad) {
  using C = phi::dtype::complex<T>;
  ctx.template Alloc<C>(x_grad);
  auto norm_type = funcs::get_norm_from_string(normalization, forward);

  funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;
  fft_r2c_func(ctx, out_grad, x_grad, axes, norm_type, !forward);

  const int64_t double_length =
      out_grad.dims()[axes.back()] - x_grad->dims()[axes.back()];
  const phi::DDim strides = phi::stride(x_grad->dims());

#if defined(__NVCC__) || defined(__HIPCC__)
  const thrust::device_vector<int64_t> strides_g(phi::vectorize(strides));
  const int64_t* pstrides = thrust::raw_pointer_cast(strides_g.data());
#else
  const int64_t* pstrides = strides.Get();
#endif

  funcs::FFTFillConjGradFunctor<C> func(
      x_grad->data<C>(), axes.back(), pstrides, double_length);
  size_t limit = x_grad->numel();
  funcs::ForRange<Context> for_range(ctx, limit);
  for_range(func);
}
}  // namespace phi
