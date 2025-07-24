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
#include "paddle/phi/kernels/complex_grad_kernel.h"

#include "fft/cuComplex.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace xfft_internal::xpu {
int combine_as_complex(int N, float* real, float* imag, float2* out);
int complex_spilt_float(int N, float2* in, float* real, float* imag);
}  // namespace xfft_internal::xpu

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& dev_ctx,
                        std::vector<int> shape,
                        T fill_value) {
  DenseTensor ret;
  ret.Resize(common::make_ddim(shape));
  dev_ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(dev_ctx, &ret, fill_value);
  return ret;
}

template <typename T, typename Context>
void RealGradKernel(const Context& dev_ctx,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  if (dx && dx->numel() == 0) {
    dev_ctx.template Alloc<T>(dx);
    return;
  }
  auto numel = dout.numel();
  auto* dx_data =
      dev_ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));
  DenseTensor imag = Fill<phi::dtype::Real<T>, Context>(
      dev_ctx, common::vectorize<int>(dout.dims()), phi::dtype::Real<T>(0.0));
  int r = xfft_internal::xpu::combine_as_complex(
      numel,
      const_cast<phi::dtype::Real<T>*>(dout.data<phi::dtype::Real<T>>()),
      imag.data<phi::dtype::Real<T>>(),
      reinterpret_cast<cuFloatComplex*>(dx_data));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

template <typename T, typename Context>
void ImagGradKernel(const Context& dev_ctx,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  if (dx && dx->numel() == 0) {
    dev_ctx.template Alloc<T>(dx);
    return;
  }
  auto numel = dout.numel();
  auto* dx_data =
      dev_ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));
  DenseTensor real = Fill<phi::dtype::Real<T>, Context>(
      dev_ctx, common::vectorize<int>(dout.dims()), phi::dtype::Real<T>(0.0));
  int r = xfft_internal::xpu::combine_as_complex(
      numel,
      real.data<phi::dtype::Real<T>>(),
      const_cast<phi::dtype::Real<T>*>(dout.data<phi::dtype::Real<T>>()),
      reinterpret_cast<cuFloatComplex*>(dx_data));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

template <typename T, typename Context>
void ComplexGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  using C = phi::dtype::complex<T>;
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dy->dims())), 0, dy);
      }
    }
    return;
  }
  auto numel = dout.numel();
  DenseTensor real_dout, imag_dout;
  real_dout.Resize(dout.dims());
  imag_dout.Resize(dout.dims());
  T* real_data = dev_ctx.template Alloc<T>(&real_dout);
  T* imag_data = dev_ctx.template Alloc<T>(&imag_dout);

  int r = xfft_internal::xpu::complex_spilt_float(
      numel,
      reinterpret_cast<cuFloatComplex*>(const_cast<C*>(dout.data<C>())),
      real_data,
      imag_data);
  PADDLE_ENFORCE_XPU_SUCCESS(r);

  if (dx) {
    if (x.dims() == dout.dims()) {
      dx->ShareDataWith(real_dout);
    } else {
      ExpandGradKernel<T, Context>(
          dev_ctx, x, real_dout, phi::IntArray(phi::vectorize(x.dims())), dx);
    }
  }

  if (dy) {
    if (y.dims() == dout.dims()) {
      dy->ShareDataWith(imag_dout);
    } else {
      ExpandGradKernel<T, Context>(
          dev_ctx, y, imag_dout, phi::IntArray(phi::vectorize(y.dims())), dy);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(imag_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ImagGradKernel,
                   phi::dtype::complex<float>) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(real_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::RealGradKernel,
                   phi::dtype::complex<float>) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(
    complex_grad, XPU, ALL_LAYOUT, phi::ComplexGradKernel, float) {
  kernel->InputAt(2).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif  // PADDLE_WITH_XPU_FFT
