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

#ifdef PADDLE_WITH_XPU_FFT
#include <string>
#include <vector>

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fft_grad_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj_xpu.h"
#include "paddle/phi/kernels/pad_kernel.h"

namespace phi {
template <typename T, typename Context>
void FFTC2CGradKernel(const Context& dev_ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad && x_grad->numel() == 0) {
    return;
  }
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;
  fft_c2c_func(dev_ctx, out_grad, x_grad, axes, norm_type, !forward);
}

template <typename T, typename Context>
void FFTR2CGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      bool onesided,
                      DenseTensor* x_grad) {
  using R = typename T::value_type;
  DenseTensor complex_x_grad = EmptyLike<T>(dev_ctx, x);
  dev_ctx.template Alloc<R>(x_grad);
  if (x_grad && x_grad->numel() == 0) {
    return;
  }
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;

  if (!onesided) {
    fft_c2c_func(dev_ctx, out_grad, &complex_x_grad, axes, norm_type, !forward);
  } else {
    DenseTensor full_dy;
    DenseTensorMeta full_dy_meta(out_grad.type(), x_grad->dims());
    full_dy.set_meta(full_dy_meta);
    auto zero_length = static_cast<int>(full_dy.dims().at(axes.back()) -
                                        out_grad.dims().at(axes.back()));
    auto rank = out_grad.dims().size();
    std::vector<int> pads(rank * 2, 0);
    pads[axes.back() * 2 + 1] = zero_length;
    PadKernel<T>(dev_ctx, out_grad, pads, static_cast<float>(0.0), &full_dy);
    fft_c2c_func(dev_ctx, full_dy, &complex_x_grad, axes, norm_type, !forward);
  }
  RealKernel<T>(dev_ctx, complex_x_grad, x_grad);
}

template <typename T, typename Context>
void FFTC2RGradKernel(const Context& dev_ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      int64_t last_dim_size UNUSED,
                      DenseTensor* x_grad) {
  using C = phi::dtype::complex<T>;
  dev_ctx.template Alloc<C>(x_grad);
  if (x_grad && x_grad->numel() == 0) {
    return;
  }
  auto norm_type = funcs::get_norm_from_string(normalization, forward);
  funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;
  fft_r2c_func(dev_ctx, out_grad, x_grad, axes, norm_type, !forward);
  // Manually double interior elements along the Hermitian axis. The XFFT
  // library's FFTFillConjGrad kernel can crash on XPU (kl3ChannelCheckErrors),
  // so we build a 2D float weight [1, n] and elementwise-multiply the
  // real and imaginary parts of x_grad. We combine via a fresh tensor
  // to avoid ComplexKernel crash when x_grad is both input and output.
  const int64_t last_axis = axes.back();
  const int64_t n = x_grad->dims()[last_axis];
  if (n > 2) {
    // Build 2D float weight [1, 1], [1, n-2], [1, 1] -> concat -> [1, n]
    // for direct broadcasting with x_real [..., n] without needing Resize.
    DenseTensor w_dc = phi::Full<float>(dev_ctx, {1, 1}, 1.0f);
    DenseTensor w_interior = phi::Full<float>(dev_ctx, {1, n - 2}, 2.0f);
    DenseTensor w_nyquist = phi::Full<float>(dev_ctx, {1, 1}, 1.0f);
    std::vector<const DenseTensor*> w_tensors = {
        &w_dc, &w_interior, &w_nyquist};
    DenseTensor w_2d;
    phi::ConcatKernel<float, Context>(
        dev_ctx, w_tensors, phi::Scalar(1), &w_2d);
    // Split x_grad into real/imag float parts (creates FRESH tensors)
    DenseTensor x_real = phi::Real<C, Context>(dev_ctx, *x_grad);
    DenseTensor x_imag = phi::Imag<C, Context>(dev_ctx, *x_grad);
    // Multiply real and imag parts by the weight.
    // NOTE: MultiplyKernel requires output metadata to be pre-set.
    DenseTensor real_out;
    real_out.Resize(x_real.dims());
    dev_ctx.template Alloc<float>(&real_out);
    phi::MultiplyKernel<float, Context>(dev_ctx, x_real, w_2d, &real_out);
    DenseTensor imag_out;
    imag_out.Resize(x_imag.dims());
    dev_ctx.template Alloc<float>(&imag_out);
    phi::MultiplyKernel<float, Context>(dev_ctx, x_imag, w_2d, &imag_out);
    // Combine into a FRESH tensor using ComplexKernel (avoids crash when
    // x_grad is both input and output of ComplexKernel), then copy to x_grad.
    // NOTE: ComplexKernel also requires output metadata to be pre-set.
    DenseTensor combined =
        phi::EmptyLike<C, Context>(dev_ctx, real_out);
    phi::ComplexKernel<float, Context>(dev_ctx, real_out, imag_out, &combined);
    // Copy combined to x_grad via assignment operator.
    *x_grad = combined;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    fft_c2c_grad, XPU, ALL_LAYOUT, phi::FFTC2CGradKernel, phi::complex64) {}
PD_REGISTER_KERNEL(
    fft_c2r_grad, XPU, ALL_LAYOUT, phi::FFTC2RGradKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
PD_REGISTER_KERNEL(
    fft_r2c_grad, XPU, ALL_LAYOUT, phi::FFTR2CGradKernel, phi::complex64) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#endif