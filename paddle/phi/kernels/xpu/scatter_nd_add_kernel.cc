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

#include "paddle/phi/kernels/scatter_nd_add_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ScatterNdAddKernel(const Context &ctx,
                        const DenseTensor &x,
                        const DenseTensor &index,
                        const DenseTensor &updates,
                        DenseTensor *out) {
  const T *x_ptr = x.data<T>();
  const T *updates_ptr = updates.data<T>();

  T *out_ptr = ctx.template Alloc<T>(out);
  int r = xpu::copy(ctx.x_context(), x_ptr, out_ptr, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  if (updates.numel() == 0) return;

  if (index.numel() == 0) {
    int loop_time =
        static_cast<int>(index.dims().size() == 0 ? 1 : index.dims()[0]);

    for (int i = 0; i < loop_time; i++) {
      // xpu::add only support float or float16 template typename
      // now, register this op only with float type
      r = xpu::add<T>(ctx.x_context(),
                      updates_ptr + out->numel() * i,
                      out_ptr,
                      out_ptr,
                      out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    }
    return;
  }

  const phi::DataType index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto x_shape = phi::vectorize<int64_t>(x.dims());
  auto index_shape = phi::vectorize<int64_t>(index.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int64_t> x_vec = {
      x_shape.data(), static_cast<int>(x_shape.size()), nullptr};

  int index_size = static_cast<int>(index.numel());

  if (index_type == phi::DataType::INT32) {
    auto index_data = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> index_vec{nullptr, index_size, index_data};
    r = xpu::scatter_nd<T, int>(ctx.x_context(),
                                nullptr,
                                updates_ptr,
                                out_ptr,
                                index_vec,
                                x_vec,
                                index_shape,
                                false);
  } else {
    auto index_data = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> index_vec{nullptr, index_size, index_data};
    r = xpu::scatter_nd<T, int64_t>(ctx.x_context(),
                                    nullptr,
                                    updates_ptr,
                                    out_ptr,
                                    index_vec,
                                    x_vec,
                                    index_shape,
                                    false);
  }

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_nd_add");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    scatter_nd_add, XPU, ALL_LAYOUT, phi::ScatterNdAddKernel, float) {}
