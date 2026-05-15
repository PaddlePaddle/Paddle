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
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj_xpu.h"

namespace phi {
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

  // XPU FFT library requires all axes to have > 8 elements.
  // For small axes, fall back to CPU FFT.
  bool need_cpu_fallback = false;
  for (auto axis : axes) {
    if (x.dims()[axis] <= 8) {
      need_cpu_fallback = true;
      break;
    }
  }
  if (need_cpu_fallback) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));
    DenseTensor cpu_x;
    cpu_x.Resize(x.dims());
    (*cpu_ctx).template Alloc<T>(&cpu_x);
    phi::Copy(dev_ctx, x, cpu_place, true, &cpu_x);
    DenseTensor cpu_out;
    cpu_out.Resize(out->dims());
    (*cpu_ctx).template Alloc<T>(&cpu_out);
    funcs::FFTC2CFunctor<CPUContext, T, T> fft_c2c_cpu_func;
    fft_c2c_cpu_func(*cpu_ctx, cpu_x, &cpu_out, axes, norm_type, forward);
    phi::Copy(dev_ctx, cpu_out, dev_ctx.GetPlace(), false, out);
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

  // XPU FFT library requires all axes to have > 8 elements.
  // For small axes, fall back to CPU FFT.
  bool need_cpu_fallback = false;
  for (auto axis : axes) {
    if (x.dims()[axis] <= 8) {
      need_cpu_fallback = true;
      break;
    }
  }
  if (need_cpu_fallback) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));
    DenseTensor cpu_x;
    cpu_x.Resize(x.dims());
    (*cpu_ctx).template Alloc<T>(&cpu_x);
    phi::Copy(dev_ctx, x, cpu_place, true, &cpu_x);
    DenseTensor cpu_out;
    cpu_out.Resize(out->dims());
    (*cpu_ctx).template Alloc<R>(&cpu_out);
    funcs::FFTC2RFunctor<CPUContext, T, R> fft_c2r_cpu_func;
    fft_c2r_cpu_func(*cpu_ctx, cpu_x, &cpu_out, axes, norm_type, forward);
    phi::Copy(dev_ctx, cpu_out, dev_ctx.GetPlace(), false, out);
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

  // XPU FFT requires all axes to have > 8 elements (Problem 1).
  // XPU legacy cufftPlanMany API has precision issues with
  // multi-dimensional R2C transformations (Problem 2).
  // For these cases, fall back to CPU FFT.
  bool need_cpu_fallback = false;
  for (auto axis : axes) {
    if (x.dims()[axis] <= 8) {
      need_cpu_fallback = true;
      break;
    }
  }
  if (!need_cpu_fallback && axes.size() > 1) {
    need_cpu_fallback = true;
  }
  if (need_cpu_fallback) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));
    DenseTensor cpu_x;
    cpu_x.Resize(x.dims());
    (*cpu_ctx).template Alloc<T>(&cpu_x);
    phi::Copy(dev_ctx, x, cpu_place, true, &cpu_x);
    if (onesided) {
      DenseTensor cpu_out;
      cpu_out.Resize(out->dims());
      (*cpu_ctx).template Alloc<C>(&cpu_out);
      funcs::FFTR2CFunctor<CPUContext, T, C> fft_r2c_cpu_func;
      fft_r2c_cpu_func(*cpu_ctx, cpu_x, &cpu_out, axes, norm_type, forward);
      phi::Copy(dev_ctx, cpu_out, dev_ctx.GetPlace(), false, out);
    } else {
      // For non-onesided: compute onesided R2C on CPU,
      // then fill conjugate on XPU.
      DDim onesided_out_shape = x.dims();
      const int64_t last_fft_axis = axes.back();
      const int64_t onesided_last_axis_size =
          out->dims().at(last_fft_axis) / 2 + 1;
      onesided_out_shape[last_fft_axis] = onesided_last_axis_size;
      DenseTensor cpu_onesided_out;
      cpu_onesided_out.Resize(vectorize(onesided_out_shape));
      (*cpu_ctx).template Alloc<C>(&cpu_onesided_out);
      funcs::FFTR2CFunctor<CPUContext, T, C> fft_r2c_cpu_func;
      fft_r2c_cpu_func(
          *cpu_ctx, cpu_x, &cpu_onesided_out, axes, norm_type, forward);
      DenseTensor xpu_onesided_out =
          Empty<C, Context>(dev_ctx, vectorize(onesided_out_shape));
      phi::Copy(dev_ctx,
                cpu_onesided_out,
                dev_ctx.GetPlace(),
                false,
                &xpu_onesided_out);
      funcs::FFTFillConj<Context, C>(dev_ctx, &xpu_onesided_out, out, axes);
    }
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
