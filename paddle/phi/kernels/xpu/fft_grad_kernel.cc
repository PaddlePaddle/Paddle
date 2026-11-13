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
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj_xpu.h"
#include "paddle/phi/kernels/pad_kernel.h"

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

  if (NeedsCpuFallback(out_grad.dims(), axes)) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

    DenseTensor out_grad_cpu;
    out_grad_cpu.Resize(out_grad.dims());
    cpu_ctx->template Alloc<T>(&out_grad_cpu);
    Copy(dev_ctx, out_grad, cpu_place, false, &out_grad_cpu);
    x_grad->Resize(x_grad->dims());
    DenseTensor x_grad_cpu;
    x_grad_cpu.Resize(x_grad->dims());
    cpu_ctx->template Alloc<T>(&x_grad_cpu);

    funcs::FFTC2CFunctor<CPUContext, T, T> fft_c2c_func;
    fft_c2c_func(
        *cpu_ctx, out_grad_cpu, &x_grad_cpu, axes, norm_type, !forward);

    Copy(dev_ctx, x_grad_cpu, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

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

  if (NeedsCpuFallback(out_grad.dims(), axes)) {
    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));

    // out_grad is the upstream gradient (complex64). Copy it to CPU.
    DenseTensor out_grad_cpu;
    out_grad_cpu.Resize(out_grad.dims());
    cpu_ctx->template Alloc<T>(&out_grad_cpu);
    Copy(dev_ctx, out_grad, cpu_place, false, &out_grad_cpu);

    // Derive shapes from x_grad rather than from x
    // (x may not have memory allocated in the backward context).
    DenseTensor complex_x_grad_cpu =
        EmptyLike<T>(*cpu_ctx, *x_grad);
    x_grad->Resize(x_grad->dims());
    DenseTensor x_grad_cpu;
    x_grad_cpu.Resize(x_grad->dims());
    cpu_ctx->template Alloc<R>(&x_grad_cpu);

    funcs::FFTC2CFunctor<CPUContext, T, T> fft_c2c_func;

    if (!onesided) {
      fft_c2c_func(
          *cpu_ctx, out_grad_cpu, &complex_x_grad_cpu, axes, norm_type, !forward);
    } else {
      DenseTensor full_dy_cpu;
      DenseTensorMeta full_dy_meta(out_grad_cpu.type(), x_grad_cpu.dims());
      full_dy_cpu.set_meta(full_dy_meta);
      auto zero_length = static_cast<int>(full_dy_cpu.dims().at(axes.back()) -
                                          out_grad_cpu.dims().at(axes.back()));
      auto rank = out_grad_cpu.dims().size();
      std::vector<int> pads(rank * 2, 0);
      pads[axes.back() * 2 + 1] = zero_length;
      PadKernel<T>(*cpu_ctx,
                   out_grad_cpu,
                   pads,
                   static_cast<float>(0.0),
                   &full_dy_cpu);
      fft_c2c_func(
          *cpu_ctx, full_dy_cpu, &complex_x_grad_cpu, axes, norm_type, !forward);
    }
    RealKernel<T>(*cpu_ctx, complex_x_grad_cpu, &x_grad_cpu);

    Copy(dev_ctx, x_grad_cpu, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

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
  funcs::FFTFillConjGrad<Context, C>(dev_ctx, out_grad, axes, x_grad);
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