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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/legacy/gpu/moe_fuse_op.h"
#include "paddle/phi/kernels/legacy/gpu/moe_ops_utils.h"

namespace phi {

template <typename T, typename Context>
void apply_moe_dispatch_fwd(const Context &dev_ctx,
                            const T *x,
                            const float *gate_logits,
                            const float *corr_bias,
                            int64_t num_rows,
                            int64_t num_experts,
                            int64_t hidden_size,
                            int64_t capacity,
                            int64_t k,
                            T *y,
                            float *combine_weights,
                            int *scatter_index,
                            int64_t *expert_offset,
                            int *expert_id,
                            bool use_pad,
                            cudaStream_t stream) {
  int *permuted_rows = nullptr;
  int *permuted_experts = nullptr;
  topk_gating(dev_ctx,
              x,
              gate_logits,
              corr_bias,
              &permuted_rows,
              &permuted_experts,
              num_rows,
              num_experts,
              hidden_size,
              capacity,
              k,
              combine_weights,
              scatter_index,
              expert_offset,
              expert_id,
              use_pad,
              stream);

  initialize_moe_routing_kernelLauncher(x,
                                        y,
                                        permuted_rows,
                                        scatter_index,
                                        permuted_experts,
                                        expert_offset,
                                        combine_weights,
                                        static_cast<int>(num_rows),
                                        static_cast<int>(hidden_size),
                                        static_cast<int>(k),
                                        capacity,
                                        use_pad,
                                        stream);

  return;
}

template <typename T, typename Context>
void moe_dispatch_fwd(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &gate_logits,
                      const paddle::optional<DenseTensor> &corr_bias,
                      int64_t num_rows,
                      int64_t num_experts,
                      int64_t hidden_size,
                      int64_t capacity,
                      int64_t k,
                      const DenseTensor &y,
                      const DenseTensor &combine_weights,
                      const DenseTensor &scatter_index,
                      const DenseTensor &expert_offset,
                      const DenseTensor &expert_id,
                      bool use_pad) {
  apply_moe_dispatch_fwd<T, Context>(
      dev_ctx,
      x.data<T>(),
      gate_logits.data<float>(),
      corr_bias ? corr_bias.get_ptr()->data<float>() : nullptr,
      num_rows,
      num_experts,
      hidden_size,
      capacity,
      k,
      const_cast<T *>(y.data<T>()),
      const_cast<float *>(combine_weights.data<float>()),
      const_cast<int *>(scatter_index.data<int>()),
      const_cast<int64_t *>(expert_offset.data<int64_t>()),
      const_cast<int *>(expert_id.data<int>()),
      use_pad,
      dev_ctx.stream());
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

  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(y->dims())), 0, y);
  auto x_dims = x.dims();
  auto gate_logits_dims = gate_logits.dims();

  const int64_t num_rows = x_dims[0];
  const int64_t hidden_size = x_dims[1];
  const int64_t num_experts = gate_logits_dims[1];

  moe_dispatch_fwd<T, Context>(dev_ctx,
                               x,
                               gate_logits,
                               corr_bias,
                               num_rows,
                               num_experts,
                               hidden_size,
                               capacity,
                               k,
                               *y,
                               *combine_weights,
                               *scatter_index,
                               *expert_offset,
                               *expert_id,
                               use_pad);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeGradDispatchKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
