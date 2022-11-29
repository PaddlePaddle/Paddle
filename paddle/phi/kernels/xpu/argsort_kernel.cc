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

#include "paddle/phi/kernels/argsort_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   DenseTensor* output,
                   DenseTensor* indices) {
  auto in_dims = input.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  int n = in_dims[axis];

  auto input_data = input.data<T>();
  auto output_data = dev_ctx.template Alloc<T>(output);
  auto indices_data = dev_ctx.template Alloc<int64_t>(indices);

  bool is_need_transpose = true;
  if (axis == -1 || axis + 1 == in_dims.size()) {
    is_need_transpose = false;
  }
  int len_before = phi::product(phi::slice_ddim(in_dims, 0, axis));
  int len_after =
      phi::product(phi::slice_ddim(in_dims, axis + 1, in_dims.size()));
  int m = len_before * len_after;
  int len = m * n;
  std::vector<int> permute_vec{0, 2, 1};
  std::vector<int> data_shape{len_before, n, len_after};
  std::vector<int> data_shape_trans{len_before, len_after, n};

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (is_need_transpose) {
    T* input_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(input_data_trans);
    T* output_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(output_data_trans);
    int64_t* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(indices_data_trans);

    int r = xpu::transpose<T>(dev_ctx.x_context(),
                              input_data,
                              input_data_trans,
                              data_shape,
                              permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    input_data = input_data_trans;
    output_data = output_data_trans;
    indices_data = indices_data_trans;
  }

  int ret = xpu::sort<T, int64_t>(dev_ctx.x_context(),
                                  input_data,
                                  output_data,
                                  indices_data,
                                  m,
                                  n,
                                  descending);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sort");

  if (is_need_transpose) {
    int r = xpu::transpose<T>(dev_ctx.x_context(),
                              output_data,
                              output->data<T>(),
                              data_shape_trans,
                              permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    r = xpu::transpose<int64_t>(dev_ctx.x_context(),
                                indices_data,
                                indices->data<int64_t>(),
                                data_shape_trans,
                                permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    argsort, XPU, ALL_LAYOUT, phi::ArgsortKernel, float, int, int64_t) {}
