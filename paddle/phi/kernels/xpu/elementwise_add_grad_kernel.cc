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

#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {
template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   int axis,
                   DenseTensor* dx,
                   DenseTensor* dy) {
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
  using XPUType = typename XPUTypeTrait<T>::Type;
  funcs::ElementwiseGradPreProcess(dout, dx);
  auto* dz = &dout;
  const DDim& dz_dims = dz->dims();

  const T* dz_data = dz->data<T>();

  if (dx != nullptr) {
    T* dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dz_dims) {
      if (dx_data != dz_data) {
        int ret = xpu::copy(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(dz_data),
                            reinterpret_cast<XPUType*>(dx->data<T>()),
                            dx->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dz, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(*dz)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(dx->dims(), dz_dims, axis);
      std::vector<int64_t> dz_vector = common::vectorize<int64_t>(dz_dims);

      int ret = xpu::reduce_sum<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(dz_data),
          reinterpret_cast<XPUType*>(dx->data<T>()),
          dz_vector,
          std::vector<int64_t>(reduce_dims.begin(), reduce_dims.end()));
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
    }
  }

  if (dy != nullptr) {
    T* dy_data = dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dz_dims) {
      if (dy_data != dz_data) {
        int ret = xpu::copy(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(dz_data),
                            reinterpret_cast<XPUType*>(dy->data<T>()),
                            dy->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(dy->dims(), dz_dims, axis);
      std::vector<int64_t> dz_vector = common::vectorize<int64_t>(dz_dims);
      int ret = xpu::reduce_sum<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(dz_data),
          reinterpret_cast<XPUType*>(dy_data),
          dz_vector,
          std::vector<int64_t>(reduce_dims.begin(), reduce_dims.end()));
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
    }
  }
}
#ifdef PADDLE_WITH_XPU_FFT
template <>
void AddGradKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const DenseTensor& dout,
    int axis,
    DenseTensor* dx,
    DenseTensor* dy) {
  using T = phi::dtype::complex<float>;
  const bool compute_dx = (dx != nullptr);
  const bool compute_dy = (dy != nullptr);

  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
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

    AddGradKernel<float, XPUContext>(dev_ctx,
                                     tmp_real,  // unused
                                     tmp_imag,  // unused
                                     dout_real,
                                     axis,
                                     compute_dx ? &dx_real : nullptr,
                                     compute_dy ? &dy_real : nullptr);

    AddGradKernel<float, XPUContext>(dev_ctx,
                                     tmp_real,  // unused
                                     tmp_imag,  // unused
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
#endif
}  // namespace phi

PD_REGISTER_KERNEL(add_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   float,
                   int,
                   int64_t) {
}
