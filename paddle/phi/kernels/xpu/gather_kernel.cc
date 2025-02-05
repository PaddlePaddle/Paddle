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
void GatherKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& index,
                  const Scalar& axis,
                  DenseTensor* out) {
  auto axis_v = axis.to<int64_t>();
  if (axis_v < 0) {
    axis_v += static_cast<int64_t>(x.dims().size());
  }
  const auto& index_type = index.dtype();

  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || index.numel() == 0) return;

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        common::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        common::errors::InvalidArgument(
            "The index should be 0D, 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }
  std::vector<int64_t> xshape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    xshape[i] = x.dims()[i];
  }

  using XPUType = typename XPUTypeTrait<T>::Type;

  int r = XPU_SUCCESS;
  if (index_type == DataType::INT32) {
    r = xpu::paddle_gather<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v);
  } else {
    r = xpu::paddle_gather<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
}

}  // namespace phi

PD_REGISTER_KERNEL(gather,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   bool) {}
