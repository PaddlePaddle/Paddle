// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/fft_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj_xpu.h"

namespace phi {

// XPU FFT requires all signal dimensions > 8. The XPU cuFFT library
// crashes with heap corruption for smaller dimensions, so we fall
// back to CPU for those cases.
static bool NeedsCpuFallback(const DDim& dims,
                             const std::vector<int64_t>& axes) {
  for (auto axis : axes) {
    if (dims[axis] <= 8) return true;
  }
  return false;
}

template <typename T, typename Context>
void FFTC2CKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (NeedsCpuFallback(x.dims(), axes)) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

    DenseTensor x_cpu, out_cpu;
    x_cpu.Resize(x.dims());
    cpu_ctx->template Alloc<T>(&x_cpu);
    Copy(dev_ctx, x, cpu_place, false, &x_cpu);
    out_cpu.Resize(out->dims());
    cpu_ctx->template Alloc<T>(&out_cpu);

    funcs::FFTC2CFunctor<CPUContext, T, T> fft_c2c_func;
    fft_c2c_func(*cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);

    Copy(dev_ctx, out_cpu, dev_ctx.GetPlace(), false, out);
    return;
  }

  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;
  fft_c2c_func(dev_ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTC2RKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  int64_t last_dim_size UNUSED,
                  DenseTensor* out) {
  using R = typename T::value_type;  // get real type
  dev_ctx.template Alloc<R>(out);
  if (x.numel() == 0) {
    Full<R, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (NeedsCpuFallback(x.dims(), axes)) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

    DenseTensor x_cpu;
    x_cpu.Resize(x.dims());
    cpu_ctx->template Alloc<T>(&x_cpu);
    Copy(dev_ctx, x, cpu_place, false, &x_cpu);
    out->Resize(out->dims());
    DenseTensor out_cpu;
    out_cpu.Resize(out->dims());
    cpu_ctx->template Alloc<R>(&out_cpu);

    funcs::FFTC2RFunctor<CPUContext, T, R> fft_c2r_func;
    fft_c2r_func(*cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);

    Copy(dev_ctx, out_cpu, dev_ctx.GetPlace(), false, out);
    return;
  }

  funcs::FFTC2RFunctor<Context, T, R> fft_c2r_func;
  fft_c2r_func(dev_ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTR2CKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  bool onesided,
                  DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  dev_ctx.template Alloc<C>(out);
  if (x.numel() == 0) {
    Full<C, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (NeedsCpuFallback(x.dims(), axes)) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

    DenseTensor x_cpu;
    x_cpu.Resize(x.dims());
    cpu_ctx->template Alloc<T>(&x_cpu);
    Copy(dev_ctx, x, cpu_place, false, &x_cpu);
    out->Resize(out->dims());
    DenseTensor out_cpu;
    out_cpu.Resize(out->dims());
    cpu_ctx->template Alloc<C>(&out_cpu);

    if (onesided) {
      funcs::FFTR2CFunctor<CPUContext, T, C> fft_r2c_func;
      fft_r2c_func(*cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);
    } else {
      // R2C produces onesided output; fill the conjugate-symmetric half
      // manually since FFTFillConj<CPUContext> is not available in this
      // translation unit (it conflicts with fft_fill_conj_xpu.h).
      DDim onesided_shape = x_cpu.dims();
      const int64_t last_fft_axis = axes.back();
      const int64_t out_last_dim = out->dims().at(last_fft_axis);
      const int64_t onesided_last_dim = out_last_dim / 2 + 1;
      onesided_shape[last_fft_axis] = onesided_last_dim;
      DenseTensor onesided_out;
      onesided_out.Resize(onesided_shape);
      cpu_ctx->template Alloc<C>(&onesided_out);

      funcs::FFTR2CFunctor<CPUContext, T, C> fft_r2c_func2;
      fft_r2c_func2(
          *cpu_ctx, x_cpu, &onesided_out, axes, norm_type, forward);

      // Fill conjugate-symmetric half using the same algorithm as
      // FFTFillConjFunctor (fft_fill_conj.h) but inlined to avoid
      // header conflicts with fft_fill_conj_xpu.h.
      const auto* src_data = onesided_out.data<C>();
      auto* dst_data = out_cpu.data<C>();
      const DDim& dst_dims = out_cpu.dims();
      const auto dst_strides = common::stride(dst_dims);
      const auto src_strides = common::stride(onesided_shape);
      const int64_t nrank = static_cast<int64_t>(dst_dims.size());
      std::vector<bool> is_fft_axis(nrank, false);
      for (auto a : axes) is_fft_axis[a] = true;

      for (int64_t idx = 0; idx < out_cpu.numel(); ++idx) {
        // Check if idx is in the conjugate-symmetric half of last axis
        int64_t q = idx;
        int64_t pos_on_last = 0;
        for (int64_t d = 0; d < nrank; ++d) {
          if (d == last_fft_axis) {
            pos_on_last = q / dst_strides[d];
          }
          q = q % dst_strides[d];
        }
        if (pos_on_last < onesided_last_dim) {
          // Direct copy: map dst index to src index using src strides
          int64_t src_idx = 0;
          int64_t r = idx;
          for (int64_t d = 0; d < nrank; ++d) {
            int64_t p = r / dst_strides[d];
            r = r % dst_strides[d];
            src_idx += p * src_strides[d];
          }
          dst_data[idx] = src_data[src_idx];
        } else {
          // Conjugate-symmetric: reflect on all FFT axes
          int64_t src_idx = 0;
          int64_t r = idx;
          for (int64_t d = 0; d < nrank; ++d) {
            int64_t p = r / dst_strides[d];
            r = r % dst_strides[d];
            if (is_fft_axis[d] && p != 0) {
              p = dst_dims[d] - p;
            }
            src_idx += p * src_strides[d];
          }
          auto src_val = src_data[src_idx];
          dst_data[idx] = C(src_val.real, -src_val.imag);
        }
      }
    }

    Copy(dev_ctx, out_cpu, dev_ctx.GetPlace(), false, out);
    return;
  }

  funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;

  if (onesided) {
    fft_r2c_func(dev_ctx, x, out, axes, norm_type, forward);
  } else {
    DDim onesided_out_shape = x.dims();
    const int64_t last_fft_axis = axes.back();
    const int64_t onesided_last_axis_size =
        out->dims().at(last_fft_axis) / 2 + 1;
    onesided_out_shape[last_fft_axis] = onesided_last_axis_size;
    DenseTensor onesided_out =
        Empty<C, Context>(dev_ctx, vectorize(onesided_out_shape));
    fft_r2c_func(dev_ctx, x, &onesided_out, axes, norm_type, forward);
    funcs::FFTFillConj<Context, C>(dev_ctx, &onesided_out, out, axes);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    fft_c2c, XPU, ALL_LAYOUT, phi::FFTC2CKernel, phi::complex64) {}
PD_REGISTER_KERNEL(
    fft_c2r, XPU, ALL_LAYOUT, phi::FFTC2RKernel, phi::complex64) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
PD_REGISTER_KERNEL(fft_r2c, XPU, ALL_LAYOUT, phi::FFTR2CKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif