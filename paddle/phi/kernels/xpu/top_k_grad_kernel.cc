// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/top_k_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/xpu/xpu_mem_util.h"

namespace {
inline void GetDims(
    const phi::DDim& dim, int axis, int64_t* pre, int64_t* n, int64_t* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}
}  // namespace

namespace phi {

template <typename T, typename Context>
void TopkGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    const DenseTensor& out_grad,
                    const Scalar& k_scalar,
                    int axis,
                    bool largest,
                    bool sorted,
                    DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  const auto& in_dims = x.dims();
  int k = k_scalar.to<int>();

  // get the real the axis and the k
  if (axis < 0) {
    axis += in_dims.size();
  }

  // allocate the xpu memory for the x_grad
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* out_grad_data = out_grad.data<T>();
  int64_t* indices_data = const_cast<int64_t*>(indices.data<int64_t>());

  if (in_dims.size() == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  int64_t pre, n, post;
  GetDims(in_dims, axis, &pre, &n, &post);

  FullKernel<T, Context>(dev_ctx,
                         common::vectorize(x_grad->dims()),
                         0.0f,
                         x_grad->dtype(),
                         x_grad);

  // We want to add a different constant value to each row of a 2D matrix for
  // the following scatter kernel. For example: If n = 5 and the input matrix
  // is:
  // [[0, 1, 2],
  //  [3, 4, 5],
  //  [6, 7, 8]]
  //
  // Then:
  // Row 1 → add 0 * 5 = 0
  // Row 2 → add 1 * 5 = 5
  // Row 3 → add 2 * 5 = 10
  //
  // Result:
  // [[ 0,  1,  2],
  //  [ 8,  9, 10],
  //  [16, 17, 18]]

  PADDLE_ENFORCE_EQ(indices.numel(),
                    pre * k * post,
                    ::common::errors::InvalidArgument(
                        "indices.numel() must be pre * k * post, but got "
                        "indices.numel() = %ld, pre "
                        "= %ld, k = %d, n = %ld, post = %ld, axis=%ld",
                        indices.numel(),
                        pre,
                        k,
                        n,
                        post,
                        axis));

  int64_t* offsets = RAII_GUARD.alloc<int64_t>(pre * post);
  int64_t* accmulated_indices = RAII_GUARD.alloc<int64_t>(indices.numel());

  int ret =
      xpu::range<int64_t>(dev_ctx.x_context(), offsets, 0LL, n, pre * post);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "range");

  ret = xpu::broadcast_mul<int64_t>(dev_ctx.x_context(),
                                    indices_data,
                                    offsets,
                                    accmulated_indices,
                                    {pre, k, post},
                                    {pre, 1, post});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_mul");

  const xpu::VectorParam<int64_t> indices_vp = {
      nullptr, indices.numel(), accmulated_indices};
  // launch the xpu kernel to assign the grad
  ret = xpu::scatter<XPUType, int64_t>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_grad_data),
      reinterpret_cast<const XPUType*>(out_grad_data),
      reinterpret_cast<XPUType*>(x_grad_data),
      indices_vp,
      common::vectorize(x_grad->dims()),
      axis,
      /*is_overwrite*/ true);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "scatter");
}

template <typename T, typename Context>
void TopkV1GradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      const Scalar& k_scalar,
                      DenseTensor* x_grad) {
  TopkGradKernel<T, Context>(
      dev_ctx, x, indices, out_grad, k_scalar, -1, true, true, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(topk_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TopkGradKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(topk_v1_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TopkV1GradKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
