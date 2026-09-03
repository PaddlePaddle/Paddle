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

#include "paddle/phi/kernels/cast_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

#ifdef PADDLE_WITH_XPU_FFT
template <class T, class Context>
static DenseTensor Fill(const Context& dev_ctx,
                        std::vector<int> shape,
                        T fill_value) {
  DenseTensor ret;
  ret.Resize(shape);
  dev_ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(dev_ctx, &ret, fill_value);
  return ret;
}
#endif

template <typename InT, typename OutT, typename Context>
void CastXPUKernelImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  using XPUInT = typename XPUTypeTrait<InT>::Type;
  using XPUOutT = typename XPUTypeTrait<OutT>::Type;

  auto numel = x.numel();
  if (numel == 0) {
    dev_ctx.template Alloc<OutT>(out);
    return;
  }

  const auto* in_data = x.data<InT>();
  auto* out_data = dev_ctx.template Alloc<OutT>(out);

  if (std::is_same<InT, OutT>::value) {
    int ret = xpu::copy(dev_ctx.x_context(),
                        reinterpret_cast<const int8_t*>(in_data),
                        reinterpret_cast<int8_t*>(out_data),
                        x.numel() * phi::SizeOf(x.dtype()));
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
    return;
  }

  if (std::is_same<InT, dtype::bfloat16>::value &&
          !std::is_same<OutT, float>::value ||
      !std::is_same<InT, float>::value &&
          std::is_same<OutT, dtype::bfloat16>::value) {
    // bfloat -> non float, or non float -> bfloat, use float buffer
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float* cast_buffer = RAII_GUARD.alloc_l3_or_gm<float>(numel);
    // step 1: InT to float
    int r = xpu::cast<XPUInT, float>(dev_ctx.x_context(),
                                     reinterpret_cast<const XPUInT*>(in_data),
                                     cast_buffer,
                                     numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    // step 2: float to OutT
    r = xpu::cast<float, XPUOutT>(dev_ctx.x_context(),
                                  cast_buffer,
                                  reinterpret_cast<XPUOutT*>(out_data),
                                  numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    return;
  }

  int r = xpu::cast<XPUInT, XPUOutT>(dev_ctx.x_context(),
                                     reinterpret_cast<const XPUInT*>(in_data),
                                     reinterpret_cast<XPUOutT*>(out_data),
                                     numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
}

#ifdef PADDLE_WITH_XPU_FFT
template <typename InT, typename RealT, typename Context>
void CastRealToComplexXPUKernelImpl(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<phi::dtype::complex<RealT>>(out);
    return;
  }

  DenseTensor real;
  real.Resize(x.dims());
  CastXPUKernelImpl<InT, RealT, Context>(dev_ctx, x, &real);
  DenseTensor imag = Fill<RealT, Context>(
      dev_ctx, vectorize<int>(x.dims()), static_cast<RealT>(0.0));
  phi::ComplexKernel<RealT>(dev_ctx, real, imag, out);
}

template <typename InT, typename OutRealT, typename Context>
void CastComplexToComplexXPUKernelImpl(const Context& dev_ctx,
                                       const DenseTensor& x,
                                       DenseTensor* out) {
  DenseTensor real = Real<InT, Context>(dev_ctx, x);
  DenseTensor imag = Imag<InT, Context>(dev_ctx, x);
  DenseTensor out_real;
  DenseTensor out_imag;
  out_real.Resize(x.dims());
  out_imag.Resize(x.dims());
  CastXPUKernelImpl<dtype::Real<InT>, OutRealT, Context>(
      dev_ctx, real, &out_real);
  CastXPUKernelImpl<dtype::Real<InT>, OutRealT, Context>(
      dev_ctx, imag, &out_imag);
  phi::ComplexKernel<OutRealT>(dev_ctx, out_real, out_imag, out);
}
#endif

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  if (x.dtype() == out_dtype) {
    if (x.dims() == make_ddim({-1})) {
      *out = x;
      return;
    }
    if (!out->IsSharedWith(x)) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  switch (out_dtype) {
    case DataType::INT32:
      CastXPUKernelImpl<T, int, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT32:
      CastXPUKernelImpl<T, float, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT16:
      CastXPUKernelImpl<T, dtype::float16, Context>(dev_ctx, x, out);
      break;
    case DataType::BFLOAT16:
      CastXPUKernelImpl<T, dtype::bfloat16, Context>(dev_ctx, x, out);
      break;
    case DataType::INT64:
      CastXPUKernelImpl<T, int64_t, Context>(dev_ctx, x, out);
      break;
    case DataType::BOOL:
      CastXPUKernelImpl<T, bool, Context>(dev_ctx, x, out);
      break;
    case DataType::INT8:
      CastXPUKernelImpl<T, int8_t, Context>(dev_ctx, x, out);
      break;
    case DataType::UINT8:
      CastXPUKernelImpl<T, uint8_t, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT64:
      CastXPUKernelImpl<T, double, Context>(dev_ctx, x, out);
      break;
    case DataType::INT16:
      CastXPUKernelImpl<T, int16_t, Context>(dev_ctx, x, out);
      break;
#ifdef PADDLE_WITH_XPU_FFT
    case DataType::COMPLEX64:
      CastRealToComplexXPUKernelImpl<T, float, Context>(dev_ctx, x, out);
      break;
    case DataType::COMPLEX128:
      CastRealToComplexXPUKernelImpl<T, double, Context>(dev_ctx, x, out);
      break;
#endif
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Not supported cast %d -> %d", x.dtype(), out_dtype));
  }
}
#ifdef PADDLE_WITH_XPU_FFT
template <>
void CastKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                            const DenseTensor& x,
                                            DataType out_dtype,
                                            DenseTensor* out) {
  using T = phi::complex64;
  if (x.dtype() == out_dtype) {
    if (x.dims() == make_ddim({-1})) {
      *out = x;
      return;
    }
    if (!out->IsSharedWith(x)) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  if (out_dtype == DataType::COMPLEX128) {
    CastComplexToComplexXPUKernelImpl<T, double, XPUContext>(dev_ctx, x, out);
    return;
  }
  DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  CastKernel<float, XPUContext>(dev_ctx, x_real, out_dtype, out);
}

template <>
void CastKernel<phi::complex128, XPUContext>(const XPUContext& dev_ctx,
                                             const DenseTensor& x,
                                             DataType out_dtype,
                                             DenseTensor* out) {
  using T = phi::complex128;
  if (x.dtype() == out_dtype) {
    if (x.dims() == make_ddim({-1})) {
      *out = x;
      return;
    }
    if (!out->IsSharedWith(x)) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  if (out_dtype == DataType::COMPLEX64) {
    CastComplexToComplexXPUKernelImpl<T, float, XPUContext>(dev_ctx, x, out);
    return;
  }
  DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  CastKernel<double, XPUContext>(dev_ctx, x_real, out_dtype, out);
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(cast,
                   XPU,
                   ALL_LAYOUT,
                   phi::CastKernel,
                   int16_t,
                   int32_t,
                   float,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
                   phi::complex128,
#endif
                   int64_t,
                   bool,
                   int8_t,
                   uint8_t,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
