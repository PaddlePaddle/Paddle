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

#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

#ifdef PADDLE_WITH_XPU_FFT
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#endif

namespace phi {

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      int axis,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  funcs::ElementwiseGradPreProcess(dout, dx);

  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_div_grad<XPUType>(
        xpu_ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  XPUElementwiseGrad<T, XPUType>(dev_ctx, x, y, dout, axis, dx, dy, f, true);
}

#ifdef PADDLE_WITH_XPU_FFT
// Complex64 divide grad specialization: XPU does not have a native complex64
// divide_grad kernel, so we implement complex division gradients using
// real/imag decomposition.
// For complex division out = x / y:
//   dx = dout / conj(y) => dx_real = (dr*yr + di*yi) / (yr^2+yi^2), dx_imag =
//   (di*yr - dr*yi) / (yr^2+yi^2) dy = -dout * conj(out/y) => dy_real = -(dr*or
//   + di*oi) / (yr^2+yi^2), dy_imag = -(di*or - dr*oi) / (yr^2+yi^2)
// where conj denotes complex conjugate and out/y uses the same denominator as
// forward pass.
template <>
void DivideGradKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                                  const DenseTensor& x,
                                                  const DenseTensor& y,
                                                  const DenseTensor& out,
                                                  const DenseTensor& dout,
                                                  int axis,
                                                  DenseTensor* dx,
                                                  DenseTensor* dy) {
  using T = phi::complex64;
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        Full<T, XPUContext>(dev_ctx, dx->dims(), T(0), dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        Full<T, XPUContext>(dev_ctx, dy->dims(), T(0), dy);
      }
    }
    return;
  }
  funcs::ElementwiseGradPreProcess(dout, dx);

  DenseTensor dout_real = Real<T, XPUContext>(dev_ctx, dout);
  DenseTensor dout_imag = Imag<T, XPUContext>(dev_ctx, dout);
  DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
  DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);

  // Common denominator: y_real^2 + y_imag^2
  DenseTensor denom = Add<float, XPUContext>(
      dev_ctx,
      Multiply<float, XPUContext>(dev_ctx, y_real, y_real),
      Multiply<float, XPUContext>(dev_ctx, y_imag, y_imag));

  if (dx) {
    // dx = dout / conj(y)
    // dx_real = (dout_real * y_real + dout_imag * y_imag) / denom
    // dx_imag = (dout_imag * y_real - dout_real * y_imag) / denom
    DenseTensor dx_real = Divide<float, XPUContext>(
        dev_ctx,
        Add<float, XPUContext>(
            dev_ctx,
            Multiply<float, XPUContext>(dev_ctx, dout_real, y_real),
            Multiply<float, XPUContext>(dev_ctx, dout_imag, y_imag)),
        denom);
    DenseTensor dx_imag = Divide<float, XPUContext>(
        dev_ctx,
        Subtract<float, XPUContext>(
            dev_ctx,
            Multiply<float, XPUContext>(dev_ctx, dout_imag, y_real),
            Multiply<float, XPUContext>(dev_ctx, dout_real, y_imag)),
        denom);
    dev_ctx.template Alloc<T>(dx);
    if (x.dims() == dout.dims()) {
      phi::ComplexKernel<float>(dev_ctx, dx_real, dx_imag, dx);
    } else {
      DenseTensor dx_real_expanded, dx_imag_expanded;
      dx_real_expanded.Resize(dx->dims());
      dx_imag_expanded.Resize(dx->dims());
      ExpandGradKernel<float, XPUContext>(dev_ctx,
                                          x,
                                          dx_real,
                                          phi::IntArray(vectorize(x.dims())),
                                          &dx_real_expanded);
      ExpandGradKernel<float, XPUContext>(dev_ctx,
                                          x,
                                          dx_imag,
                                          phi::IntArray(vectorize(x.dims())),
                                          &dx_imag_expanded);
      phi::ComplexKernel<float>(
          dev_ctx, dx_real_expanded, dx_imag_expanded, dx);
    }
  }

  if (dy) {
    DenseTensor out_real = Real<T, XPUContext>(dev_ctx, out);
    DenseTensor out_imag = Imag<T, XPUContext>(dev_ctx, out);

    // dy = -dout * conj(out/y)
    // For complex: dy_real = -(dout_real * out_real + dout_imag * out_imag) /
    // denom
    //              dy_imag = -(dout_imag * out_real - dout_real * out_imag) /
    //              denom
    // Use Scalar(-1.0f) to negate dout before multiplying
    DenseTensor neg_dout_real = Multiply<float, XPUContext>(
        dev_ctx,
        dout_real,
        Full<float, XPUContext>(dev_ctx,
                                phi::IntArray(vectorize(dout.dims())),
                                phi::Scalar(-1.0f)));
    DenseTensor neg_dout_imag = Multiply<float, XPUContext>(
        dev_ctx,
        dout_imag,
        Full<float, XPUContext>(dev_ctx,
                                phi::IntArray(vectorize(dout.dims())),
                                phi::Scalar(-1.0f)));

    DenseTensor dy_real = Divide<float, XPUContext>(
        dev_ctx,
        Add<float, XPUContext>(
            dev_ctx,
            Multiply<float, XPUContext>(dev_ctx, neg_dout_real, out_real),
            Multiply<float, XPUContext>(dev_ctx, neg_dout_imag, out_imag)),
        denom);

    DenseTensor dy_imag = Divide<float, XPUContext>(
        dev_ctx,
        Subtract<float, XPUContext>(
            dev_ctx,
            Multiply<float, XPUContext>(dev_ctx, neg_dout_imag, out_real),
            Multiply<float, XPUContext>(dev_ctx, neg_dout_real, out_imag)),
        denom);

    dev_ctx.template Alloc<T>(dy);
    if (y.dims() == dout.dims()) {
      phi::ComplexKernel<float>(dev_ctx, dy_real, dy_imag, dy);
    } else {
      DenseTensor dy_real_expanded, dy_imag_expanded;
      dy_real_expanded.Resize(dy->dims());
      dy_imag_expanded.Resize(dy->dims());
      ExpandGradKernel<float, XPUContext>(dev_ctx,
                                          y,
                                          dy_real,
                                          phi::IntArray(vectorize(y.dims())),
                                          &dy_real_expanded);
      ExpandGradKernel<float, XPUContext>(dev_ctx,
                                          y,
                                          dy_imag,
                                          phi::IntArray(vectorize(y.dims())),
                                          &dy_imag_expanded);
      phi::ComplexKernel<float>(
          dev_ctx, dy_real_expanded, dy_imag_expanded, dy);
    }
  }
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(divide_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DivideGradKernel,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
#endif
                   float) {
}
