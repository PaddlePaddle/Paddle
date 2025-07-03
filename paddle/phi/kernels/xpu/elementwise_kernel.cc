//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_XPU_FFT
#include "fft/cuComplex.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
namespace xfft_internal::xpu {
int RemainderFunctor(int N, float2* input_x, float2* input_y, float2* output);
}
#endif

namespace phi {

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_mod<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void RemainderKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    DenseTensor* out) {
  using T = phi::dtype::complex<float>;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = phi::funcs::BroadcastTwoDims(x_dims, y_dims);
  std::vector<int64_t> out_dims_vec = phi::vectorize(out_dims);

  auto complex_expand = [](const XPUContext& dev_ctx,
                           const DenseTensor& x,
                           const std::vector<int64_t>& out_dims_vec,
                           DenseTensor* out) {
    DenseTensor real_out, imag_out;
    real_out.Resize(out->dims());
    imag_out.Resize(out->dims());
    dev_ctx.template Alloc<float>(&real_out);
    dev_ctx.template Alloc<float>(&imag_out);
    const DenseTensor real = Real<T, XPUContext>(dev_ctx, x);
    const DenseTensor imag = Imag<T, XPUContext>(dev_ctx, x);
    ExpandKernel<float, XPUContext>(
        dev_ctx, real, phi::IntArray(out_dims_vec), &real_out);
    ExpandKernel<float, XPUContext>(
        dev_ctx, imag, phi::IntArray(out_dims_vec), &imag_out);
    phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
  };

  DenseTensor broadcasted_x, broadcasted_y;
  T* x_data = nullptr;
  T* y_data = nullptr;

  if (x_dims == out_dims) {
    x_data = const_cast<T*>(x.data<T>());
  } else {
    broadcasted_x.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_x);
    complex_expand(dev_ctx, x, out_dims_vec, &broadcasted_x);
    x_data = broadcasted_x.data<T>();
  }

  if (y_dims == out_dims) {
    y_data = const_cast<T*>(y.data<T>());
  } else {
    broadcasted_y.Resize(out_dims);
    dev_ctx.template Alloc<T>(&broadcasted_y);
    complex_expand(dev_ctx, y, out_dims_vec, &broadcasted_y);
    y_data = broadcasted_y.data<T>();
  }

  dev_ctx.template Alloc<T>(out);
  int r = xfft_internal::xpu::RemainderFunctor(
      out->numel(),
      reinterpret_cast<cuFloatComplex*>(x_data),
      reinterpret_cast<cuFloatComplex*>(y_data),
      reinterpret_cast<cuFloatComplex*>(out->data<T>()));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(floor_divide,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(maximum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(minimum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(remainder,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   phi::dtype::float16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   int32_t,
                   int64_t) {
}
PD_REGISTER_KERNEL(elementwise_pow,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
