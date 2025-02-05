// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/scatter_nd_add_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ScatterNdAddGradKernel(const Context &ctx,
                            const DenseTensor &index,
                            const DenseTensor &updates,
                            const DenseTensor &out_grad,
                            DenseTensor *x_grad,
                            DenseTensor *updates_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int ret = xpu::SUCCESS;
  const T *out_grad_data = out_grad.data<T>();
  if (x_grad) {
    auto *x_grad_data = ctx.template Alloc<T>(x_grad);
    ret = xpu::copy<XPUType>(ctx.x_context(),
                             reinterpret_cast<const XPUType *>(out_grad_data),
                             reinterpret_cast<XPUType *>(x_grad_data),
                             out_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
  }

  if (updates_grad) {
    auto *updates_grad_data = ctx.template Alloc<T>(updates_grad);
    if (updates_grad->numel() == 0) {
      return;
    }
    if (index.numel() == 0) {
      auto index_dims = index.dims();
      auto index_dims_size = index_dims.size();
      int64_t end_size = index_dims[index_dims_size - 1];
      PADDLE_ENFORCE_EQ(
          end_size,
          0,
          errors::InvalidArgument(
              "Size of the last dim of the index tensor [%d] should be 0",
              end_size));
      auto remain_dims = common::slice_ddim(index_dims, 0, index_dims_size - 1);
      int64_t remain_numel = common::product(remain_dims);
      int64_t updates_grad_numel = updates_grad->numel();
      int64_t out_grad_numel = out_grad.numel();
      PADDLE_ENFORCE_EQ(
          remain_numel * out_grad_numel,
          updates_grad_numel,
          errors::InvalidArgument("out_grad numel[%d] * remain numel[%d] "
                                  "should math updates_grad numel[%d]",
                                  out_grad_numel,
                                  remain_numel,
                                  updates_grad_numel));
      ret = xpu::broadcast<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType *>(out_grad_data),
          reinterpret_cast<XPUType *>(updates_grad_data),
          {1, out_grad_numel},
          {remain_numel, out_grad_numel});
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
      return;
    }

    auto index_shape_vec = common::vectorize<int64_t>(index.dims());
    if (index_shape_vec.size() == 1) {
      index_shape_vec.insert(index_shape_vec.begin(), 1);
    }
    auto out_grad_shape_vec = common::vectorize<int64_t>(out_grad.dims());
    xpu::VectorParam<int64_t> out_grad_shape_param = {
        out_grad_shape_vec.data(),
        static_cast<int64_t>(out_grad_shape_vec.size()),
        nullptr};

    if (index.dtype() == DataType::INT32) {
      ret = xpu::gather_nd<XPUType, int>(
          ctx.x_context(),
          reinterpret_cast<const XPUType *>(out_grad_data),
          index.data<int>(),
          reinterpret_cast<XPUType *>(updates_grad_data),
          out_grad_shape_param,
          index_shape_vec);
    } else {
      ret = xpu::gather_nd<XPUType, int64_t>(
          ctx.x_context(),
          reinterpret_cast<const XPUType *>(out_grad_data),
          index.data<int64_t>(),
          reinterpret_cast<XPUType *>(updates_grad_data),
          out_grad_shape_param,
          index_shape_vec);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather_nd");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(scatter_nd_add_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScatterNdAddGradKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
