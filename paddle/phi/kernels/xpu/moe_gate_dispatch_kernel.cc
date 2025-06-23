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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace xpu = baidu::xpu::api;

namespace phi {

template <typename T, typename Context>
void moe_dispatch_fwd(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &gate_logits,
                      const paddle::optional<DenseTensor> &corr_bias,
                      int64_t capacity,
                      int64_t k,
                      DenseTensor *y,
                      DenseTensor *combine_weights,
                      DenseTensor *scatter_index,
                      DenseTensor *expert_offset,
                      DenseTensor *expert_id,
                      bool use_pad) {
  PADDLE_ENFORCE_EQ(gate_logits.dtype(),
                    paddle::DataType::FLOAT32,
                    ::common::errors::InvalidArgument(
                        "Unsupported dtype for gate_logits, "
                        "currently only float32 is supported."));

  int64_t s = x.dims()[0];
  int64_t d = x.dims()[1];
  int64_t e = gate_logits.dims()[1];

  PADDLE_ENFORCE_GT(
      k,
      0,
      ::common::errors::InvalidArgument("the k of topk must more than 0."));
  PADDLE_ENFORCE_GT(capacity,
                    0,
                    ::common::errors::InvalidArgument(
                        "the capacity of each expert must more than 0."));
  PADDLE_ENFORCE_GE(e,
                    k,
                    ::common::errors::InvalidArgument(
                        "the amount of experts must greater than k."));
  PADDLE_ENFORCE_EQ(
      corr_bias.is_initialized(),
      false,
      ::common::errors::InvalidArgument("corr_bias is not supported yet"));

  using XPUType = typename XPUTypeTrait<T>::Type;

  // xpu input data
  auto x_data = reinterpret_cast<const XPUType *>(x.data<T>());
  auto gate_logits_data =
      reinterpret_cast<const float *>(gate_logits.data<float>());
  // xpu output data
  auto y_data = reinterpret_cast<XPUType *>(y->data<T>());
  auto combine_weights_data =
      reinterpret_cast<float *>(combine_weights->data<float>());
  auto scatter_index_data = reinterpret_cast<int *>(scatter_index->data<int>());
  auto expert_offset_data =
      reinterpret_cast<int64_t *>(expert_offset->data<int64_t>());
  auto expert_id_data = reinterpret_cast<int *>(expert_id->data<int>());
  // xpu interface
  auto ret = xpu::moe_dispatch<XPUType>(dev_ctx.x_context(),
                                        x_data,
                                        gate_logits_data,
                                        s,
                                        d,
                                        k,
                                        e,
                                        capacity,
                                        y_data,
                                        combine_weights_data,
                                        scatter_index_data,
                                        expert_offset_data,
                                        expert_id_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "moe_dispatch");
}

template <typename T, typename Context>
void MoeGradDispatchKernel(const Context &dev_ctx,
                           const DenseTensor &x,
                           const DenseTensor &gate_logits,
                           const paddle::optional<DenseTensor> &corr_bias,
                           const int64_t k,
                           const int64_t capacity,
                           const bool use_pad,
                           DenseTensor *y,
                           DenseTensor *combine_weights,
                           DenseTensor *scatter_index,
                           DenseTensor *expert_offset,
                           DenseTensor *expert_id) {
  dev_ctx.template Alloc<int>(expert_id);
  dev_ctx.template Alloc<int64_t>(expert_offset);
  dev_ctx.template Alloc<int>(scatter_index);
  dev_ctx.template Alloc<float>(combine_weights);
  dev_ctx.template Alloc<T>(y);
  PD_CHECK(use_pad);  // only support use_pad=true

  moe_dispatch_fwd<T, Context>(dev_ctx,
                               x,
                               gate_logits,
                               corr_bias,
                               capacity,
                               k,
                               y,
                               combine_weights,
                               scatter_index,
                               expert_offset,
                               expert_id,
                               use_pad);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeGradDispatchKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
