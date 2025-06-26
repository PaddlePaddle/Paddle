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
#include "paddle/phi/kernels/full_kernel.h"

namespace xpu = baidu::xpu::api;

namespace phi {

template <typename T, typename Context>
void MoECombineForwardKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& combine_weights,
                             const DenseTensor& scatter_index,
                             DenseTensor* y) {
  int64_t seq_len = combine_weights.dims()[0];
  int64_t k = combine_weights.dims()[1];
  int64_t hidden_size = x.dims()[1];

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto weight_data =
      reinterpret_cast<const XPUType*>(combine_weights.data<T>());
  auto index_data = scatter_index.data<int>();
  auto y_data = reinterpret_cast<XPUType*>(y->data<T>());

  int ret = xpu::moe_combine<XPUType, int>(dev_ctx.x_context(),
                                           x_data,
                                           weight_data,
                                           index_data,
                                           y_data,
                                           k,
                                           seq_len,
                                           hidden_size);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "moe_combine");
}

template <typename T, typename Context>
void MoeCombineKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& combine_weights,
                      const DenseTensor& scatter_index,
                      DenseTensor* y) {
  PD_CHECK(x.dims().size() == 2 && combine_weights.dims().size() == 2 &&
               scatter_index.dims().size() == 2,
           "The shape of X must be 2.");
  PD_CHECK(scatter_index.dtype() == paddle::DataType::INT32,
           "MoE combine only supports int32 for scatter_index!");

  dev_ctx.template Alloc<T>(y);
  MoECombineForwardKernel<T, Context>(
      dev_ctx, x, combine_weights, scatter_index, y);
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_combine,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeCombineKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
