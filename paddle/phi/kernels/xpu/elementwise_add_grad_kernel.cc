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
  using XPUType = typename XPUTypeTrait<T>::Type;
  funcs::ElementwiseGradPreProcess(dout, dx);
  auto* dz = &dout;
  const DDim& dz_dims = dz->dims();

  const T* dz_data = dz->data<T>();

  if (dx != nullptr) {
    T* dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dz_dims) {
      if (dx_data != dz_data) {
        Copy(dev_ctx, *dz, dev_ctx.GetPlace(), false, dx);
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
      std::vector<int> dz_vector = phi::vectorize<int>(dz_dims);

      int ret =
          xpu::reduce_sum<XPUType>(dev_ctx.x_context(),
                                   reinterpret_cast<const XPUType*>(dz_data),
                                   reinterpret_cast<XPUType*>(dx->data<T>()),
                                   dz_vector,
                                   reduce_dims);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
    }
  }

  if (dy != nullptr) {
    T* dy_data = dy->mutable_data<T>(dev_ctx.GetPlace());
    if (dy->dims() == dz_dims) {
      if (dy_data != dz_data) {
        Copy(dev_ctx, *dz, dev_ctx.GetPlace(), false, dy);
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(dy->dims(), dz_dims, axis);
      std::vector<int> dz_vector = phi::vectorize<int>(dz_dims);
      int ret =
          xpu::reduce_sum<XPUType>(dev_ctx.x_context(),
                                   reinterpret_cast<const XPUType*>(dz_data),
                                   reinterpret_cast<XPUType*>(dy_data),
                                   dz_vector,
                                   reduce_dims);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    add_grad, XPU, ALL_LAYOUT, phi::AddGradKernel, phi::dtype::float16, float) {
}
