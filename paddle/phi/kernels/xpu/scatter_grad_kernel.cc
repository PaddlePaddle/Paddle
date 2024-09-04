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

#include "paddle/phi/kernels/scatter_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ScatterGradKernel(const Context &ctx,
                       const DenseTensor &index,
                       const DenseTensor &updates,
                       const DenseTensor &out_grad,
                       bool overwrite,
                       DenseTensor *x_grad,
                       DenseTensor *updates_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "scatter_op index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s]",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  T *x_grad_data = nullptr;
  T *updates_grad_data = nullptr;
  if (x_grad != nullptr) {
    ctx.template Alloc<T>(x_grad);
    x_grad_data = x_grad->data<T>();
  }
  if (updates_grad != nullptr) {
    ctx.template Alloc<T>(updates_grad);
    updates_grad_data = updates_grad->data<T>();
  }

  std::vector<int64_t> x_grad_shape;
  DDim out_dims = out_grad.dims();
  for (int i = 0; i < out_dims.size(); i++) {
    x_grad_shape.push_back(out_dims[i]);
  }

  int index_size = index.numel();

  int r;
  if (index_type == phi::DataType::INT32) {
    auto index_data = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> indices{nullptr, index_size, index_data};
    r = xpu::scatter_grad<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(out_grad.data<T>()),
        indices,
        reinterpret_cast<XPUType *>(x_grad_data),
        reinterpret_cast<XPUType *>(updates_grad_data),
        x_grad_shape,
        overwrite);
  } else if (index_type == phi::DataType::INT64) {
    auto index_data = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> indices{nullptr, index_size, index_data};
    r = xpu::scatter_grad<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType *>(out_grad.data<T>()),
        indices,
        reinterpret_cast<XPUType *>(x_grad_data),
        reinterpret_cast<XPUType *>(updates_grad_data),
        x_grad_shape,
        overwrite);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(scatter_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScatterGradKernel,
                   float,
                   phi::dtype::float16) {}
