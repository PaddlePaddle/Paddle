// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/legacy/xpu/batch_gemm.h"
#include "paddle/phi/kernels/legacy/xpu/batched_gemm_xpu_utils.h"

namespace phi {

template <typename T, typename Context>
void BatchedGEMM(const Context &dev_ctx,
                 const DenseTensor &lhs,
                 const DenseTensor &rhs,
                 const std::vector<int64_t> &batch_sizes,
                 const bool trans_lhs,
                 const bool trans_rhs,
                 DenseTensor *output) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(output);
  if (lhs.numel() == 0 || rhs.numel() == 0) {
    return;
  }
  // We expect layout below:
  // 1. trans_lhs = false && trans_rhs = false (group forward) :
  //    [M_total, input_hidden_size] x [num_experts, input_hidden_size,
  //    output_hidden_size] output: [M_total, output_hidden_size]
  //
  // 2. trans_lhs = false && trans_rhs = true (backward for lhs_grad, or
  // specialized forward):
  //    [M_total, output_hidden_size] x [num_experts, input_hidden_size,
  //    output_hidden_size]' output: [M_total, input_hidden_size]
  //
  // 3. trans_lhs = true && trans_rhs = false (backward for rhs_grad) :
  //    [M_total, input_hidden_size]' x [M_total, output_hidden_size]
  //    output: [num_experts, input_hidden_size, output_hidden_size]
  xpu::Context *xpu_ctx = dev_ctx.x_context();
  if (!trans_lhs) {
    // Case 1 and 2: group forward or lhs_grad (input_grad)
    // Note that this case implements grouped gemm, mapping hidden_lhs to
    // hidden_out For each expert i, This case views lhs as [Mi x K] and rhs
    // as [E x K x N] or [E x N x K], so the output is [Mtotal x N], N could
    // be input_hidden_size or output_hidden_size.
    MGroupedGemmXPUFunction<T>(
        lhs, rhs, output, trans_rhs, batch_sizes, xpu_ctx);
  } else {
    // =============================================================================
    // Case 3: group backward for rhs_grad (weight_grad)
    // Note that this case implements k-grouped gemm
    // For each expert i, this case views lhs as [K x Mi] and rhs as [Mi x
    // N], so the output is [E x K x N].
    KGroupedGemmXPUFunction<T>(lhs, rhs, output, batch_sizes, xpu_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(batched_gemm,
                   XPU,
                   ALL_LAYOUT,
                   phi::BatchedGEMM,
                   float,
                   double,
                   phi::bfloat16) {}
