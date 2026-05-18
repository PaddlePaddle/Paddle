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

#include "paddle/phi/kernels/index_get_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/xpu/index_put_xpu_utils.h"

namespace phi {

template <typename T, typename Context>
void IndexGetGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(x_grad);
  int r = xpu::constant<XPUType>(dev_ctx.x_context(),
                                 reinterpret_cast<XPUType*>(x_grad->data<T>()),
                                 x_grad->numel(),
                                 static_cast<XPUType>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  if (out_grad.numel() == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices.empty()) {
    return;
  }

  auto bd_dim = funcs::BroadCastTensorsDims(int_indices);

  DenseTensor stacked_indices(DataType::INT64);
  XPUDealWithIndices<Context>(dev_ctx, int_indices, bd_dim, &stacked_indices);

  auto x_shape = vectorize<int64_t>(x.dims());
  auto index_shape = vectorize<int64_t>(stacked_indices.dims());
  if (index_shape.size() == 1) {
    index_shape.insert(index_shape.begin(), 1);
  }
  xpu::VectorParam<int64_t> x_vec = {
      x_shape.data(), static_cast<int64_t>(x_shape.size()), nullptr};

  xpu::VectorParam<int64_t> index_vec = {
      nullptr,
      stacked_indices.numel(),
      const_cast<int64_t*>(stacked_indices.data<int64_t>())};

  r = xpu::scatter_nd<XPUType, int64_t>(
      dev_ctx.x_context(),
      nullptr,
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      index_vec,
      x_vec,
      index_shape,
      false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_nd");

  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_get_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexGetGradKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}
