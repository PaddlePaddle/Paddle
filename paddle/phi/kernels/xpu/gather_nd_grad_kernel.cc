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
  ctx.template Alloc<T>(x_grad);

  int r = XPU_SUCCESS;
  T *dx_data = x_grad->data<T>();
  r = xpu::constant<T>(
      ctx.x_context(), dx_data, x_grad->numel(), static_cast<T>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  if (out_grad.numel() == 0) return;

  if (index.numel() == 0) {
    r = xpu::copy(ctx.x_context(),
                  out_grad.data<T>(),
                  x_grad->data<T>(),
                  x_grad->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  auto index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  int index_size =
      static_cast<int>(index.dims().size() == 0 ? 1 : index.dims()[0]);
  auto x_shape = phi::vectorize<int64_t>(x_grad->dims());
  auto index_shape = phi::vectorize<int64_t>(index.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int64_t> x_vec = {
      x_shape.data(), static_cast<int>(x_shape.size()), nullptr};

  DenseTensor index_cpu(index.type());
  phi::Copy(ctx, index, phi::CPUPlace(), false, &index_cpu);

  if (index_type == phi::DataType::INT32) {
    auto index_data = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> index_vec{
        index_cpu.data<int>(), index_size, index_data};
    r = xpu::scatter_nd<T, int>(ctx.x_context(),
                                nullptr,
                                out_grad.data<T>(),
                                dx_data,
                                index_vec,
                                x_vec,
                                index_shape,
                                false);
  } else {
    auto index_data = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> index_vec{
        index_cpu.data<int64_t>(), index_size, index_data};
    r = xpu::scatter_nd<T, int64_t>(ctx.x_context(),
                                    nullptr,
                                    out_grad.data<T>(),
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
                   int64_t) {}
