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
#include "paddle/phi/kernels/complex_kernel.h"

#include "fft/cuComplex.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace xfft_internal::xpu {
int combine_as_complex(int N, float* real, float* imag, float2* out);
int complex_spilt_float(int N, float2* in, float* real, float* imag);
int Conj(int N, float2* input, float2* output);
}  // namespace xfft_internal::xpu

namespace phi {
template <typename T, typename Context>
void ConjKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (std::is_same_v<T, phi::dtype::complex<float>>) {
    int r = xfft_internal::xpu::Conj(
        x.numel(),
        reinterpret_cast<cuFloatComplex*>(const_cast<T*>(x.data<T>())),
        reinterpret_cast<cuFloatComplex*>(out->data<T>()));
    PADDLE_ENFORCE_XPU_SUCCESS(r);
  } else {
    using XPUType = typename XPUCopyTypeTrait<T>::Type;
    const auto* input_data = x.data<T>();
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(input_data),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  }
}

template <typename T, typename Context>
void RealKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
  // The allocation of imag here is redundant and could be optimized.
  phi::DenseTensor imag;
  imag.Resize(x.dims());
  dev_ctx.template Alloc<phi::dtype::Real<T>>(&imag);
  int r = xfft_internal::xpu::complex_spilt_float(
      out->numel(),
      reinterpret_cast<cuFloatComplex*>(const_cast<T*>(x.data<T>())),
      out->data<phi::dtype::Real<T>>(),
      imag.data<phi::dtype::Real<T>>());
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

template <typename T, typename Context>
void ImagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
  // The allocation of ‘real’ here is redundant and could be optimized.
  phi::DenseTensor real;
  real.Resize(x.dims());
  dev_ctx.template Alloc<phi::dtype::Real<T>>(&real);
  int r = xfft_internal::xpu::complex_spilt_float(
      out->numel(),
      reinterpret_cast<cuFloatComplex*>(const_cast<T*>(x.data<T>())),
      real.data<phi::dtype::Real<T>>(),
      out->data<phi::dtype::Real<T>>());
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

template <typename T, typename Context>
void ComplexKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = phi::funcs::BroadcastTwoDims(x_dims, y_dims);
  std::vector<int64_t> out_dims_vec = phi::vectorize(out_dims);

  DenseTensor broadcasted_x, broadcasted_y;
  T* x_data = nullptr;
  T* y_data = nullptr;

  if (x_dims == out_dims) {
    x_data = const_cast<T*>(x.data<T>());
  } else {
    broadcasted_x.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_x);
    ExpandKernel<T, Context>(
        dev_ctx, x, phi::IntArray(out_dims_vec), &broadcasted_x);
    x_data = broadcasted_x.data<T>();
  }

  if (y_dims == out_dims) {
    y_data = const_cast<T*>(y.data<T>());
  } else {
    broadcasted_y.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_y);
    ExpandKernel<T, Context>(
        dev_ctx, y, phi::IntArray(out_dims_vec), &broadcasted_y);
    y_data = broadcasted_y.data<T>();
  }

  dev_ctx.template Alloc<C>(out);
  int r = xfft_internal::xpu::combine_as_complex(
      out->numel(),
      x_data,
      y_data,
      reinterpret_cast<cuFloatComplex*>(out->data<C>()));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}
}  // namespace phi

PD_REGISTER_KERNEL(conj,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConjKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>) {}

PD_REGISTER_KERNEL(
    real, XPU, ALL_LAYOUT, phi::RealKernel, phi::dtype::complex<float>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(
    imag, XPU, ALL_LAYOUT, phi::ImagKernel, phi::dtype::complex<float>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(complex, XPU, ALL_LAYOUT, phi::ComplexKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif  // PADDLE_WITH_XPU_FFT
