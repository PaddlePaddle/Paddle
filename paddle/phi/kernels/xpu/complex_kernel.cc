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
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace xfft_internal::xpu {
// just for declaration here, the real implementation is in libcufft.so
template <typename T, typename TComplex>
int combine_as_complex(int N, const T* real, const T* imag, TComplex* out);
template <>
int combine_as_complex(int N,
                       const float* real,
                       const float* imag,
                       float2* out);
template <>
int combine_as_complex(int N,
                       const double* real,
                       const double* imag,
                       double2* out);

template <typename TComplex, typename T>
int complex_spilt(int N, const TComplex* in, T* real, T* imag);
template <>
int complex_spilt(int N, const float2* in, float* real, float* imag);
template <>
int complex_spilt(int N, const double2* in, double* real, double* imag);

template <typename T>  // T supports float2, double2
int Conj(int N, const T* input, T* output);
}  // namespace xfft_internal::xpu

namespace phi {
template <typename T, typename Context>
void ConjKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  dev_ctx.template Alloc<T>(out);
  if (std::is_same_v<T, phi::complex64>) {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(dev_ctx.x_context()->xpu_stream));
    int r = xfft_internal::xpu::Conj(
        x.numel(),
        reinterpret_cast<const cuFloatComplex*>(x.data<T>()),
        reinterpret_cast<cuFloatComplex*>(out->data<T>()));
    PADDLE_ENFORCE_XPU_SUCCESS(r);
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
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
  if (out->numel() == 0) {
    dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
    return;
  }
  dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
  // The allocation of imag here is redundant and could be optimized.
  DenseTensor imag;
  imag.Resize(x.dims());
  dev_ctx.template Alloc<phi::dtype::Real<T>>(&imag);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(dev_ctx.x_context()->xpu_stream));
  int r = xfft_internal::xpu::complex_spilt(
      out->numel(),
      reinterpret_cast<const cuFloatComplex*>(x.data<T>()),
      out->data<phi::dtype::Real<T>>(),
      imag.data<phi::dtype::Real<T>>());
  PADDLE_ENFORCE_XPU_SUCCESS(r);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
}

template <typename T, typename Context>
void ImagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
    return;
  }
  dev_ctx.template Alloc<phi::dtype::Real<T>>(out);
  // The allocation of ‘real’ here is redundant and could be optimized.
  DenseTensor real;
  real.Resize(x.dims());
  dev_ctx.template Alloc<phi::dtype::Real<T>>(&real);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(dev_ctx.x_context()->xpu_stream));
  int r = xfft_internal::xpu::complex_spilt(
      out->numel(),
      reinterpret_cast<const cuFloatComplex*>(x.data<T>()),
      real.data<phi::dtype::Real<T>>(),
      out->data<phi::dtype::Real<T>>());
  PADDLE_ENFORCE_XPU_SUCCESS(r);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
}

template <typename T, typename Context>
void ComplexKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<C>(out);
    return;
  }
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = funcs::BroadcastTwoDims(x_dims, y_dims);
  std::vector<int64_t> out_dims_vec = phi::vectorize(out_dims);

  DenseTensor broadcasted_x, broadcasted_y;
  const T* x_data = nullptr;
  const T* y_data = nullptr;

  if (x_dims == out_dims) {
    x_data = x.data<T>();
  } else {
    broadcasted_x.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_x);
    ExpandKernel<T, Context>(
        dev_ctx, x, phi::IntArray(out_dims_vec), &broadcasted_x);
    x_data = broadcasted_x.data<T>();
  }

  if (y_dims == out_dims) {
    y_data = y.data<T>();
  } else {
    broadcasted_y.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_y);
    ExpandKernel<T, Context>(
        dev_ctx, y, phi::IntArray(out_dims_vec), &broadcasted_y);
    y_data = broadcasted_y.data<T>();
  }

  dev_ctx.template Alloc<C>(out);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(dev_ctx.x_context()->xpu_stream));
  int r = xfft_internal::xpu::combine_as_complex(
      out->numel(),
      x_data,
      y_data,
      reinterpret_cast<cuFloatComplex*>(out->data<C>()));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
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
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64) {}

PD_REGISTER_KERNEL(real, XPU, ALL_LAYOUT, phi::RealKernel, phi::complex64) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(imag, XPU, ALL_LAYOUT, phi::ImagKernel, phi::complex64) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(complex, XPU, ALL_LAYOUT, phi::ComplexKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif  // PADDLE_WITH_XPU_FFT
