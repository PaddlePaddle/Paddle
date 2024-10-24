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

#include "paddle/phi/kernels/gather_nd_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherNdKernel(const Context &ctx,
                    const DenseTensor &x,
                    const DenseTensor &index,
                    DenseTensor *out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);

  if (x.numel() == 0 || out->numel() == 0) return;
  if (index.dims()[0] == 0 && index.numel() == 0) return;

  if (index.numel() == 0) {
    auto index_dims = index.dims();
    auto index_dims_size = index_dims.size();
    // final dim
    int64_t end_size = index_dims[index_dims_size - 1];
    PADDLE_ENFORCE_EQ(
        end_size,
        0,
        common::errors::InvalidArgument("end_size[%d] should be 0", end_size));
    // remain dim
    auto remain_ddim = common::slice_ddim(index_dims, 0, index_dims_size - 1);
    int64_t remain_numel = common::product(remain_ddim);

    int64_t x_numel = x.numel();
    int64_t y_numel = out->numel();
    PADDLE_ENFORCE_EQ(
        x_numel * remain_numel,
        y_numel,
        common::errors::InvalidArgument(
            "x_numel[%d] * remain_numel[%d] should match y_numel[%d]",
            x_numel,
            remain_numel,
            y_numel));

    // int broadcast(Context* ctx, const T* x, T* y, const std::vector<int>&
    // xshape, const std::vector<int>& yshape)
    int r = xpu::broadcast(ctx.x_context(),
                           reinterpret_cast<const XPUType *>(x.data<T>()),
                           reinterpret_cast<XPUType *>(out->data<T>()),
                           {1, x_numel},
                           {remain_numel, x_numel});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    return;
  }

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s]",
                        index_type,
                        DataType::INT32,
                        DataType::INT64));

  auto x_shape = common::vectorize<int>(x.dims());
  auto index_shape = common::vectorize<int>(index.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int> x_vec = {
      x_shape.data(), static_cast<int>(x_shape.size()), nullptr};

  int ret = XPU_SUCCESS;
#ifndef PADDLE_WITH_XPU_PLUGIN
  if (index_type == DataType::INT32) {
    ret = xpu::gather_nd<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType *>(out->data<T>()),
        x_vec,
        index_shape);
  } else {
    ret = xpu::gather_nd<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType *>(out->data<T>()),
        x_vec,
        index_shape);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather_nd");
#else
  if (index_type == DataType::INT32) {
    ret = xpu::plugin::fast_gather_nd<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType *>(out->data<T>()),
        x_vec,
        index_shape);
  } else {
    ret = xpu::plugin::fast_gather_nd<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType *>(out->data<T>()),
        x_vec,
        index_shape);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "fast_gather_nd");
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_nd,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherNdKernel,
                   float,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
