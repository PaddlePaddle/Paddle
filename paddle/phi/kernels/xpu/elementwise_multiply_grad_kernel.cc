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

#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        Full<T, Context>(dev_ctx, dx->dims(), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        Full<T, Context>(dev_ctx, dy->dims(), 0, dy);
      }
    }
    return;
  }
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
    return xpu::broadcast_mul_grad<XPUType>(
        xpu_ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  XPUElementwiseGrad<T, XPUType>(dev_ctx, x, y, dout, axis, dx, dy, f, true);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void MultiplyGradKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                                    const DenseTensor& x,
                                                    const DenseTensor& y,
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
  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  DenseTensor dout_real = Real<T, XPUContext>(dev_ctx, dout);
  DenseTensor dout_imag = Imag<T, XPUContext>(dev_ctx, dout);

  if (dx) {
    DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
    DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);
    DenseTensor dx_real = Add<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_real, y_real),
        Multiply<float, XPUContext>(dev_ctx, dout_imag, y_imag));
    DenseTensor dx_imag = Subtract<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_imag, y_real),
        Multiply<float, XPUContext>(dev_ctx, dout_real, y_imag));
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
    DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
    DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
    DenseTensor dy_real = Add<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_real, x_real),
        Multiply<float, XPUContext>(dev_ctx, dout_imag, x_imag));
    DenseTensor dy_imag = Subtract<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_imag, x_real),
        Multiply<float, XPUContext>(dev_ctx, dout_real, x_imag));
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

template <>
void MultiplyGradKernel<phi::complex128, XPUContext>(const XPUContext& dev_ctx,
                                                     const DenseTensor& x,
                                                     const DenseTensor& y,
                                                     const DenseTensor& dout,
                                                     int axis,
                                                     DenseTensor* dx,
                                                     DenseTensor* dy) {
  using T = phi::complex128;
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
  // Complex128 gradient: dx = dout * conj(y), dy = dout * conj(x)
  // Decomposed into double real/imag parts, cast to float for XPU compute.
  DenseTensor dout_real = Real<T, XPUContext>(dev_ctx, dout);
  DenseTensor dout_imag = Imag<T, XPUContext>(dev_ctx, dout);

  DenseTensor dout_real_f = Cast<double>(dev_ctx, dout_real, DataType::FLOAT32);
  DenseTensor dout_imag_f = Cast<double>(dev_ctx, dout_imag, DataType::FLOAT32);

  if (dx) {
    DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
    DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);
    DenseTensor y_real_f = Cast<double>(dev_ctx, y_real, DataType::FLOAT32);
    DenseTensor y_imag_f = Cast<double>(dev_ctx, y_imag, DataType::FLOAT32);

    // dx_real = dout_real * y_real + dout_imag * y_imag  (conj(y) gradient)
    // dx_imag = dout_imag * y_real - dout_real * y_imag
    DenseTensor dx_real_f = Add<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_real_f, y_real_f),
        Multiply<float, XPUContext>(dev_ctx, dout_imag_f, y_imag_f));
    DenseTensor dx_imag_f = Subtract<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_imag_f, y_real_f),
        Multiply<float, XPUContext>(dev_ctx, dout_real_f, y_imag_f));

    DenseTensor dx_real = Cast<float>(dev_ctx, dx_real_f, DataType::FLOAT64);
    DenseTensor dx_imag = Cast<float>(dev_ctx, dx_imag_f, DataType::FLOAT64);

    dev_ctx.template Alloc<T>(dx);
    if (x.dims() == dout.dims()) {
      phi::ComplexKernel<double>(dev_ctx, dx_real, dx_imag, dx);
    } else {
      DenseTensor dx_real_expanded, dx_imag_expanded;
      dx_real_expanded.Resize(dx->dims());
      dx_imag_expanded.Resize(dx->dims());
      ExpandGradKernel<double, XPUContext>(dev_ctx,
                                           x,
                                           dx_real,
                                           phi::IntArray(vectorize(x.dims())),
                                           &dx_real_expanded);
      ExpandGradKernel<double, XPUContext>(dev_ctx,
                                           x,
                                           dx_imag,
                                           phi::IntArray(vectorize(x.dims())),
                                           &dx_imag_expanded);
      phi::ComplexKernel<double>(
          dev_ctx, dx_real_expanded, dx_imag_expanded, dx);
    }
  }
  if (dy) {
    DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
    DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
    DenseTensor x_real_f = Cast<double>(dev_ctx, x_real, DataType::FLOAT32);
    DenseTensor x_imag_f = Cast<double>(dev_ctx, x_imag, DataType::FLOAT32);

    // dy_real = dout_real * x_real + dout_imag * x_imag  (conj(x) gradient)
    // dy_imag = dout_imag * x_real - dout_real * x_imag
    DenseTensor dy_real_f = Add<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_real_f, x_real_f),
        Multiply<float, XPUContext>(dev_ctx, dout_imag_f, x_imag_f));
    DenseTensor dy_imag_f = Subtract<float, XPUContext>(
        dev_ctx,
        Multiply<float, XPUContext>(dev_ctx, dout_imag_f, x_real_f),
        Multiply<float, XPUContext>(dev_ctx, dout_real_f, x_imag_f));

    DenseTensor dy_real = Cast<float>(dev_ctx, dy_real_f, DataType::FLOAT64);
    DenseTensor dy_imag = Cast<float>(dev_ctx, dy_imag_f, DataType::FLOAT64);

    dev_ctx.template Alloc<T>(dy);
    if (y.dims() == dout.dims()) {
      phi::ComplexKernel<double>(dev_ctx, dy_real, dy_imag, dy);
    } else {
      DenseTensor dy_real_expanded, dy_imag_expanded;
      dy_real_expanded.Resize(dy->dims());
      dy_imag_expanded.Resize(dy->dims());
      ExpandGradKernel<double, XPUContext>(dev_ctx,
                                           y,
                                           dy_real,
                                           phi::IntArray(vectorize(y.dims())),
                                           &dy_real_expanded);
      ExpandGradKernel<double, XPUContext>(dev_ctx,
                                           y,
                                           dy_imag,
                                           phi::IntArray(vectorize(y.dims())),
                                           &dy_imag_expanded);
      phi::ComplexKernel<double>(
          dev_ctx, dy_real_expanded, dy_imag_expanded, dy);
    }
  }
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(multiply_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MultiplyGradKernel,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
                   phi::complex128,
#endif
                   float) {
}
