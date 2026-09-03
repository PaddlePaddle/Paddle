/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {
template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (dout.numel() == 0) {
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      if (dx->numel() != 0) {
        Full<T, Context>(dev_ctx, dx->dims(), 0, dx);
      }
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      if (dy->numel() != 0) {
        Full<T, Context>(dev_ctx, dy->dims(), 0, dy);
      }
    }
    return;
  }

  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_sub_grad<XPUType>(
        xpu_ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  phi::XPUElementwiseGrad<T, XPUType>(
      dev_ctx, x, y, dout, axis, dx, dy, f, false);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void SubtractGradKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                                    const DenseTensor& x,
                                                    const DenseTensor& y,
                                                    const DenseTensor& dout,
                                                    int axis,
                                                    DenseTensor* dx,
                                                    DenseTensor* dy) {
  using T = phi::complex64;
  const bool compute_dx = (dx != nullptr);
  const bool compute_dy = (dy != nullptr);

  DenseTensor dout_real = Real<T, XPUContext>(dev_ctx, dout);
  DenseTensor dout_imag = Imag<T, XPUContext>(dev_ctx, dout);

  if (compute_dx || compute_dy) {
    DenseTensor dx_real, dx_imag, dy_real, dy_imag;
    DenseTensor tmp_real, tmp_imag;

    if (compute_dx) {
      dx_real.Resize(dx->dims());
      dx_imag.Resize(dx->dims());
    }
    if (compute_dy) {
      dy_real.Resize(dy->dims());
      dy_imag.Resize(dy->dims());
    }

    SubtractGradKernel<float, XPUContext>(dev_ctx,
                                          tmp_real,
                                          tmp_imag,
                                          dout_real,
                                          axis,
                                          compute_dx ? &dx_real : nullptr,
                                          compute_dy ? &dy_real : nullptr);

    SubtractGradKernel<float, XPUContext>(dev_ctx,
                                          tmp_real,
                                          tmp_imag,
                                          dout_imag,
                                          axis,
                                          compute_dx ? &dx_imag : nullptr,
                                          compute_dy ? &dy_imag : nullptr);

    if (compute_dx) {
      dev_ctx.template Alloc<T>(dx);
      phi::ComplexKernel<float>(dev_ctx, dx_real, dx_imag, dx);
    }
    if (compute_dy) {
      dev_ctx.template Alloc<T>(dy);
      phi::ComplexKernel<float>(dev_ctx, dy_real, dy_imag, dy);
    }
  }
}

template <>
void SubtractGradKernel<phi::complex128, XPUContext>(const XPUContext& dev_ctx,
                                                     const DenseTensor& x,
                                                     const DenseTensor& y,
                                                     const DenseTensor& dout,
                                                     int axis,
                                                     DenseTensor* dx,
                                                     DenseTensor* dy) {
  using T = phi::complex128;
  const bool compute_dx = (dx != nullptr);
  const bool compute_dy = (dy != nullptr);

  DenseTensor dout_real = Real<T, XPUContext>(dev_ctx, dout);
  DenseTensor dout_imag = Imag<T, XPUContext>(dev_ctx, dout);

  // Cast double parts to float since XPU xdnn does not support
  // broadcast_sub_grad<double>; use float path with type casting.
  DenseTensor dout_real_f = Cast<float>(dev_ctx, dout_real, DataType::FLOAT32);
  DenseTensor dout_imag_f = Cast<float>(dev_ctx, dout_imag, DataType::FLOAT32);

  if (compute_dx || compute_dy) {
    DenseTensor dx_real_f, dx_imag_f, dy_real_f, dy_imag_f;
    DenseTensor tmp_real, tmp_imag;

    if (compute_dx) {
      dx_real_f.Resize(dx->dims());
      dx_imag_f.Resize(dx->dims());
    }
    if (compute_dy) {
      dy_real_f.Resize(dy->dims());
      dy_imag_f.Resize(dy->dims());
    }

    SubtractGradKernel<float, XPUContext>(dev_ctx,
                                          tmp_real,
                                          tmp_imag,
                                          dout_real_f,
                                          axis,
                                          compute_dx ? &dx_real_f : nullptr,
                                          compute_dy ? &dy_real_f : nullptr);

    SubtractGradKernel<float, XPUContext>(dev_ctx,
                                          tmp_real,
                                          tmp_imag,
                                          dout_imag_f,
                                          axis,
                                          compute_dx ? &dx_imag_f : nullptr,
                                          compute_dy ? &dy_imag_f : nullptr);

    if (compute_dx) {
      DenseTensor dx_real = Cast<double>(dev_ctx, dx_real_f, DataType::FLOAT64);
      DenseTensor dx_imag = Cast<double>(dev_ctx, dx_imag_f, DataType::FLOAT64);
      dev_ctx.template Alloc<T>(dx);
      phi::ComplexKernel<double>(dev_ctx, dx_real, dx_imag, dx);
    }
    if (compute_dy) {
      DenseTensor dy_real = Cast<double>(dev_ctx, dy_real_f, DataType::FLOAT64);
      DenseTensor dy_imag = Cast<double>(dev_ctx, dy_imag_f, DataType::FLOAT64);
      dev_ctx.template Alloc<T>(dy);
      phi::ComplexKernel<double>(dev_ctx, dy_real, dy_imag, dy);
    }
  }
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(subtract_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SubtractGradKernel,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
                   phi::complex128,
#endif
                   float) {
}
