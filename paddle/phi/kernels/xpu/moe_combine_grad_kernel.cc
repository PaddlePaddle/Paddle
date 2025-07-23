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

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace xpu = baidu::xpu::api;

namespace phi {

template <typename T, typename Context>
void MoECombineBackwardKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& combine_weights,
                              const DenseTensor& scatter_index,
                              const DenseTensor& grad_y,
                              DenseTensor* grad_x,
                              DenseTensor* grad_combine_weights_helper) {
  int64_t seq_len = combine_weights.dims()[0];
  int64_t k = combine_weights.dims()[1];
  int64_t hidden_size = x.dims()[1];

  using XPUType = typename XPUTypeTrait<T>::Type;

  auto dy_data = reinterpret_cast<const XPUType*>(grad_y.data<T>());
  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto weight_data =
      reinterpret_cast<const XPUType*>(combine_weights.data<T>());
  auto index_data = scatter_index.data<int>();
  auto dx_data = reinterpret_cast<XPUType*>(grad_x->data<T>());
  auto dw_data =
      reinterpret_cast<XPUType*>(grad_combine_weights_helper->data<T>());
  int ret =
      xpu::constant<XPUType>(dev_ctx.x_context(), dx_data, x.numel(), 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
  ret = xpu::moe_combine_grad<XPUType, int>(dev_ctx.x_context(),
                                            dy_data,
                                            x_data,
                                            weight_data,
                                            index_data,
                                            dx_data,
                                            dw_data,
                                            seq_len,
                                            k,
                                            hidden_size);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "moe_combine_grad");
}

template <typename T, typename Context>
void MoeCombineGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& combine_weights,
                          const DenseTensor& scatter_index,
                          const DenseTensor& grad_y,
                          DenseTensor* grad_x,
                          DenseTensor* grad_combine_weights_helper) {
  PD_CHECK(x.dims().size() == 2, "The shape of X must be 2.");
  PD_CHECK(scatter_index.dtype() == paddle::DataType::INT32,
           "MoE combine only supports int32 for scatter_index");
  dev_ctx.template Alloc<T>(grad_x);
  dev_ctx.template Alloc<T>(grad_combine_weights_helper);
  MoECombineBackwardKernel<T, Context>(dev_ctx,
                                       x,
                                       combine_weights,
                                       scatter_index,
                                       grad_y,
                                       grad_x,
                                       grad_combine_weights_helper);
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_combine_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeCombineGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
