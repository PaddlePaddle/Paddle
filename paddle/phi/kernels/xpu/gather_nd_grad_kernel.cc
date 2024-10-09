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

#include "paddle/phi/kernels/gather_nd_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherNdGradKernel(const Context &ctx,
                        const DenseTensor &x,
                        const DenseTensor &index,
                        const DenseTensor &out_grad,
                        DenseTensor *x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(x_grad);

  int r = XPU_SUCCESS;
  XPUType *dx_data = reinterpret_cast<XPUType *>(x_grad->data<T>());
  r = xpu::constant<XPUType>(
      ctx.x_context(), dx_data, x_grad->numel(), static_cast<XPUType>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  if (out_grad.numel() == 0) {
    return;
  }

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
    int64_t out_grad_numel = out_grad.numel();
    PADDLE_ENFORCE_EQ(
        x_numel * remain_numel,
        out_grad_numel,
        common::errors::InvalidArgument(
            "x_numel[%d] * remain_numel[%d] should match out_grad_numel[%d]",
            x_numel,
            remain_numel,
            out_grad_numel));

    // int reduce_sum(Context* ctx, const T* x, T* y, const std::vector<int>&
    // xshape, const std::vector<int>& rdims)
    int r =
        xpu::reduce_sum(ctx.x_context(),
                        reinterpret_cast<const XPUType *>(out_grad.data<T>()),
                        reinterpret_cast<XPUType *>(x_grad->data<T>()),
                        {remain_numel, x_numel},
                        {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    return;
  }

  auto index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s]",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto x_shape = common::vectorize<int64_t>(x_grad->dims());
  auto index_shape = common::vectorize<int64_t>(index.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int64_t> x_vec = {
      x_shape.data(), static_cast<int>(x_shape.size()), nullptr};

  int index_size = static_cast<int>(index.numel());
  if (index_type == phi::DataType::INT32) {
    auto index_data = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> index_vec{nullptr, index_size, index_data};
    r = xpu::scatter_nd<XPUType, int>(
        ctx.x_context(),
        nullptr,
        reinterpret_cast<const XPUType *>(out_grad.data<T>()),
        dx_data,
        index_vec,
        x_vec,
        index_shape,
        false);
  } else {
    auto index_data = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> index_vec{nullptr, index_size, index_data};
    r = xpu::scatter_nd<XPUType, int64_t>(
        ctx.x_context(),
        nullptr,
        reinterpret_cast<const XPUType *>(out_grad.data<T>()),
        dx_data,
        index_vec,
        x_vec,
        index_shape,
        false);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_nd");
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_nd_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherNdGradKernel,
                   float,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t) {}
