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

#include "paddle/phi/kernels/argsort_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ArgsortGradKernel(const Context& dev_ctx,
                       const DenseTensor& indices,
                       const DenseTensor& input,
                       const DenseTensor& out_grad,
                       int axis,
                       bool descending,
                       bool stable,
                       DenseTensor* in_grad) {
  auto in_dims = indices.dims();
  auto rank = in_dims.size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  dev_ctx.template Alloc<T>(in_grad);

  int r = xpu::constant<T>(dev_ctx.x_context(),
                           in_grad->data<T>(),
                           in_grad->numel(),
                           static_cast<T>(0.0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  if (out_grad.numel() == 0) return;

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, in_grad);
    return;
  }

  bool is_need_transpose = true;
  if (axis == -1 || axis + 1 == in_dims.size()) {
    is_need_transpose = false;
  }
  int len_before = common::product(common::slice_ddim(in_dims, 0, axis));
  int len_after =
      common::product(common::slice_ddim(in_dims, axis + 1, in_dims.size()));
  int m = len_before * len_after;
  int n = in_dims[axis];
  int len = m * n;
  std::vector<int> permute_vec{0, 2, 1};
  std::vector<int> data_shape{len_before, n, len_after};
  std::vector<int> data_shape_trans{len_before, len_after, n};

  const int64_t* indices_data = indices.data<int64_t>();
  const T* out_grad_data = out_grad.data<T>();
  T* in_grad_data = in_grad->data<T>();

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (is_need_transpose) {
    int64_t* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(indices_data_trans);
    T* out_grad_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(out_grad_data_trans);
    T* in_grad_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(in_grad_data_trans);

    r = xpu::transpose<int64_t>(dev_ctx.x_context(),
                                indices_data,
                                indices_data_trans,
                                data_shape,
                                permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    r = xpu::transpose<T>(dev_ctx.x_context(),
                          out_grad_data,
                          out_grad_data_trans,
                          data_shape,
                          permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    indices_data = indices_data_trans;
    out_grad_data = out_grad_data_trans;
    in_grad_data = in_grad_data_trans;
  }

  r = xpu::sort_grad<T, int64_t>(
      dev_ctx.x_context(), out_grad_data, indices_data, in_grad_data, m, n);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sort_grad");

  if (is_need_transpose) {
    r = xpu::transpose<T>(dev_ctx.x_context(),
                          in_grad_data,
                          in_grad->data<T>(),
                          data_shape_trans,
                          permute_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(argsort_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArgsortGradKernel,
                   float,
                   int,
                   int64_t) {}
