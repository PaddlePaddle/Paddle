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

#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace xpu = baidu::xpu::api;

namespace phi {

template <typename T, typename Context>
void moe_dispatch_grad(
    const Context& dev_ctx,
    const DenseTensor& combine_weights,       // [s, k]
    const DenseTensor& scatter_index,         // [k, s]
    const DenseTensor& expert_id,             // [s, k]
    const DenseTensor& y_grad,                // [num_experts * capacity, h]
    const DenseTensor& combine_weights_grad,  // [s, k]
    int64_t k,
    int64_t capacity,
    DenseTensor* x_grad,
    DenseTensor* gate_logits_grad) {
  if (combine_weights.dtype() != paddle::DataType::FLOAT32) {
    PD_THROW(
        "Unsupported dtype for combine_weights, "
        "currently only float32 is supported.");
  }
  if (scatter_index.dtype() != paddle::DataType::INT32) {
    PD_THROW(
        "Unsupported dtype for scatter_index, "
        "currently only int32 is supported.");
  }
  if (expert_id.dtype() != paddle::DataType::INT32) {
    PD_THROW(
        "Unsupported dtype for expert_id, "
        "currently only int32 is supported.");
  }
  if (combine_weights_grad.dtype() != paddle::DataType::FLOAT32) {
    PD_THROW(
        "Unsupported dtype for combine_weights_grad, "
        "currently only float32 is supported.");
  }
  if (!(y_grad.dtype() == paddle::DataType::FLOAT32 ||
        y_grad.dtype() == paddle::DataType::FLOAT16 ||
        y_grad.dtype() == paddle::DataType::BFLOAT16)) {
    PD_THROW(
        "Unsupported dtype for y_grad, "
        "currently float32, float16 and bfloat16 are supported.");
  }

  if (k <= 0) PD_THROW("the k of topk must more than 0.");
  if (capacity <= 0) PD_THROW("the capacity of each expert must more than 0.");

  int64_t num_experts = y_grad.dims()[0] / capacity;
  int64_t hidden_size = y_grad.dims()[1];
  int64_t num_rows = scatter_index.dims()[1];

  const std::vector<int32_t> axis = {1, 0};
  DenseTensor t_scatter_index;
  phi::Transpose<int, Context>(dev_ctx, scatter_index, axis, &t_scatter_index);

  // output
  DenseTensor x_grad_tmp =
      phi::Empty<T, Context>(dev_ctx, {num_rows, k, hidden_size});

  // ctx
  using XPUType = typename XPUTypeTrait<T>::Type;
  // xpu input data
  auto y_grad_data = reinterpret_cast<const XPUType*>(y_grad.data<T>());
  auto combine_weights_data =
      reinterpret_cast<const float*>(combine_weights.data<float>());
  auto t_scatter_index_data =
      reinterpret_cast<const int*>(t_scatter_index.data<int>());
  auto combine_weights_grad_data =
      reinterpret_cast<const float*>(combine_weights_grad.data<float>());
  auto expert_id_data = reinterpret_cast<const int*>(expert_id.data<int>());
  // xpu output data
  auto gate_logits_grad_data =
      reinterpret_cast<float*>(gate_logits_grad->data<float>());
  auto x_grad_tmp_data = reinterpret_cast<XPUType*>(x_grad_tmp.data<T>());
  auto x_grad_data = reinterpret_cast<XPUType*>(x_grad->data<T>());
  // xpu interface
  auto ret = xpu::moe_dispatch_grad<XPUType>(dev_ctx.x_context(),
                                             y_grad_data,
                                             combine_weights_data,
                                             t_scatter_index_data,
                                             combine_weights_grad_data,
                                             expert_id_data,
                                             gate_logits_grad_data,
                                             x_grad_tmp_data,
                                             num_rows,
                                             k,
                                             hidden_size,
                                             num_experts);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "moe_dispatch_grad");

  ret = xpu::reduce_sum(dev_ctx.x_context(),
                        x_grad_tmp_data,
                        x_grad_data,
                        {num_rows, k, hidden_size},
                        {1});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
}

template <typename T, typename Context>
void MoeGateDispatchGradKernel(const Context& dev_ctx,
                               const DenseTensor& combine_weights,
                               const DenseTensor& scatter_index,
                               const DenseTensor& expert_id,
                               const DenseTensor& y_grad,
                               const DenseTensor& combine_weights_grad,
                               const int64_t k,
                               const int64_t capacity,
                               const bool use_pad,
                               DenseTensor* x_grad,
                               DenseTensor* gate_logits_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<float>(gate_logits_grad);

  PD_CHECK(use_pad);  // only support use_pad=true
  moe_dispatch_grad<T, Context>(dev_ctx,
                                combine_weights,
                                scatter_index,
                                expert_id,
                                y_grad,
                                combine_weights_grad,
                                k,
                                capacity,
                                x_grad,
                                gate_logits_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
