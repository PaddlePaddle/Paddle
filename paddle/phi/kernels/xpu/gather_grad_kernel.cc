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

#include "paddle/phi/kernels/gather_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& index,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      bool overwrite,
                      DenseTensor* x_grad) {
  auto axis_v = axis.to<int>();
  const auto& index_type = index.dtype();

  if (out_grad.numel() == 0) {
    return;
  }

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        phi::errors::InvalidArgument(
            "The index should be 0D or 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }
  std::vector<int> xshape(x_grad->dims().size());
  for (int i = 0; i < x_grad->dims().size(); ++i) {
    xshape[i] = x_grad->dims()[i];
  }

  dev_ctx.template Alloc<T>(x_grad);
  using XPUType = typename XPUTypeTrait<T>::Type;

  int r = XPU_SUCCESS;
  if (index_type == DataType::INT32) {
    r = xpu::gather_grad<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(x_grad->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v,
        overwrite);
  } else {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* index_int_ptr_l3 = RAII_GUARD.alloc_l3_or_gm<int32_t>(index.numel());
    r = xpu::cast<int64_t, int32_t>(dev_ctx.x_context(),
                                    index.data<int64_t>(),
                                    index_int_ptr_l3,
                                    index.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

    r = xpu::gather_grad<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
        index_int_ptr_l3,
        reinterpret_cast<XPUType*>(x_grad->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v,
        overwrite);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherGradKernel,
                   float,
                   phi::dtype::float16) {}
