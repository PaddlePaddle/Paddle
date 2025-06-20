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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void MoeGateDispatchPartialNoSoftMaxTopkGradKernel(
    const Context& dev_ctx,
    const DenseTensor& combine_weights_out,
    const DenseTensor& scatter_index,
    const DenseTensor& scatter_index_rev,
    const DenseTensor& expert_offset,
    const DenseTensor& expert_offset_local,
    const DenseTensor& y_grad,
    const DenseTensor& combine_weights_out_grad,
    int64_t k,
    int64_t capacity,
    bool use_pad,
    int64_t expert_start_index,
    int64_t expert_end_index,
    DenseTensor* x_grad,
    DenseTensor* combine_weights_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<float>(combine_weights_grad);
  phi::Full<float, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(combine_weights_grad->dims())),
      0,
      combine_weights_grad);
  DenseTensor t_scatter_index;
  phi::Transpose<int, Context>(
      dev_ctx, scatter_index, {1, 0}, &t_scatter_index);

  int64_t num_rows = combine_weights_out.dims()[0];
  int64_t hidden_size = y_grad.dims()[1];
  int64_t num_experts = expert_offset.dims()[0];
  int64_t num_active = y_grad.dims()[0];

  using XPUDataType = typename XPUTypeTrait<T>::Type;
  int r = xpu::moe_gate_dispatch_partial_nosoftmaxtopk_grad(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUDataType*>(y_grad.data<T>()),
      combine_weights_out.data<float>(),
      t_scatter_index.data<int>(),
      combine_weights_out_grad.data<float>(),
      combine_weights_grad->data<float>(),
      reinterpret_cast<XPUDataType*>(x_grad->data<T>()),
      num_rows,
      k,
      hidden_size,
      num_experts,
      num_active);
  PADDLE_ENFORCE_XDNN_SUCCESS(r,
                              "moe_gate_dispatch_partial_nosoftmaxtopk_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_partial_nosoftmaxtopk_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchPartialNoSoftMaxTopkGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
