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

// Copyright (c) 5 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

namespace phi {

template <typename T, typename Context>
void MoeGateDispatchPartialNoSoftMaxTopkKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &combine_weights,
    const DenseTensor &expert_id,
    int64_t k,
    int64_t capacity,
    int64_t num_experts,
    bool use_pad,
    int64_t expert_start_index,
    int64_t expert_end_index,
    bool reverse_token_drop,
    DenseTensor *y,
    DenseTensor *combine_weights_out,
    DenseTensor *scatter_index,
    DenseTensor *scatter_index_rev,
    DenseTensor *expert_offset,
    DenseTensor *expert_nums_local) {
  dev_ctx.template Alloc<int32_t>(scatter_index);
  dev_ctx.template Alloc<int32_t>(scatter_index_rev);
  dev_ctx.template Alloc<int64_t>(expert_offset);
  dev_ctx.template Alloc<int64_t>(expert_nums_local);
  dev_ctx.template Alloc<float>(combine_weights_out);

  int64_t num_experts_diff = expert_end_index - expert_start_index;
  y->Resize({num_experts_diff * capacity, x.dims()[1]});
  dev_ctx.template Alloc<T>(y);

  phi::Full<int32_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(scatter_index->dims())),
      0,
      scatter_index);
  phi::Full<int32_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(scatter_index_rev->dims())),
      0,
      scatter_index_rev);
  phi::Full<int64_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(expert_offset->dims())),
      0,
      expert_offset);
  phi::Full<int64_t, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(expert_nums_local->dims())),
      0,
      expert_nums_local);
  phi::Full<float, Context>(
      dev_ctx,
      phi::IntArray(common::vectorize(combine_weights_out->dims())),
      0,
      combine_weights_out);
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(y->dims())), 0, y);

  int r = xpu::copy(dev_ctx.x_context(),
                    combine_weights.data<float>(),
                    combine_weights_out->data<float>(),
                    combine_weights_out->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  const auto &x_shape = x.dims();
  int64_t num_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];

  std::vector<int64_t> expert_offset_host(num_experts);
  using XPUDataType = typename XPUTypeTrait<T>::Type;

  r = xpu::moe_gate_dispatch_partial_nosoftmaxtopk(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUDataType *>(x.data<T>()),
      num_rows,
      num_experts,
      hidden_size,
      capacity,
      k,
      expert_start_index,
      expert_end_index,
      reverse_token_drop,
      expert_offset_host,
      reinterpret_cast<XPUDataType *>(y->data<T>()),
      combine_weights_out->data<float>(),
      scatter_index->data<int>(),
      scatter_index_rev->data<int>(),
      expert_offset->data<int64_t>(),
      expert_nums_local->data<int64_t>(),
      expert_id.data<int>(),
      use_pad);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "moe_gate_dispatch_partial_nosoftmaxtopk");

  if (use_pad) {
    // scatter_index_rev = scatter_index_rev.slice(0, num_experts_diff *
    // capacity);
    *scatter_index_rev = phi::Slice<int32_t, Context>(
        dev_ctx, *scatter_index_rev, {0}, {0}, {num_experts_diff * capacity});
  } else {
    if (expert_offset_host.back() > 0) {
      int64_t maximum_num_tokens = y->dims()[0];
      int64_t actual_num_tokens = expert_offset_host.back();
      PADDLE_ENFORCE_GE(
          maximum_num_tokens,
          actual_num_tokens,
          ::common::errors::PreconditionNotMet(
              "maximum number of tokens must be >= number of actual "
              "tokens, but got %ld < %ld",
              maximum_num_tokens,
              actual_num_tokens));

      y->Resize({expert_offset_host.back(), x.dims()[1]});

      // scatter_index_rev = scatter_index_rev.slice(0,
      // expert_offset_host.back());
      *scatter_index_rev = phi::Slice<int32_t, Context>(
          dev_ctx, *scatter_index_rev, {0}, {0}, {expert_offset_host.back()});
    } else {
      *y = phi::Empty<T, Context>(dev_ctx, {1, x_shape[1]});
      *scatter_index_rev =
          phi::Empty<int32_t, Context>(dev_ctx, {});  // special treatment
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_partial_nosoftmaxtopk,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchPartialNoSoftMaxTopkKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
