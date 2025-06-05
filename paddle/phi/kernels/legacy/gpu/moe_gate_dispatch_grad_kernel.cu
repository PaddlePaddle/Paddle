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

#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/legacy/gpu/moe_fuse_bwd_op.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T>
void apply_moe_dispatch_bwd(const T* y_grad,
                            const float* combine_weights,  // [s, k]
                            const int* scatter_index,      // [s, k]
                            const float* combine_weights_grad,
                            const int* expert_id,  // [s, k]
                            float* gate_logits_grad,
                            T* x_grad,
                            int64_t num_rows,
                            int64_t k,
                            int64_t dim,
                            int64_t num_experts,
                            int64_t capacity,
                            bool use_all2all_permute,
                            int64_t world_size,
                            int64_t num_local_experts,
                            cudaStream_t stream) {
  gather_with_mask_launcher<T>(y_grad,
                               scatter_index,
                               combine_weights,
                               x_grad,
                               num_rows,
                               k,
                               dim,
                               -1,
                               stream,
                               use_all2all_permute,
                               world_size,
                               num_local_experts,
                               capacity);

  topk_grad_with_mask_launcher<float>(combine_weights_grad,
                                      expert_id,
                                      combine_weights,
                                      gate_logits_grad,
                                      num_rows,
                                      k,
                                      num_experts,
                                      stream);
}

template <typename T, typename Context>
void moe_dispatch_bwd(const Context& dev_ctx,
                      const DenseTensor& combine_weights,  // [s, k]
                      const DenseTensor& scatter_index,    // [k, s]
                      const DenseTensor& expert_id,        // [s, k]
                      const DenseTensor& y_grad,  // [num_experts * capacity, h]
                      const DenseTensor& combine_weights_grad,  // [s, k]
                      const DenseTensor& x_grad,
                      const DenseTensor& gate_logits_grad,
                      int64_t capacity,
                      bool use_all2all_permute = false,
                      int64_t world_size = -1,
                      int64_t num_local_experts = -1) {
  int64_t num_rows = combine_weights.dims()[0];
  int64_t k = combine_weights.dims()[1];
#ifdef MOE_OPS_AUTO
  int64_t hidden_size = y_grad.dims()[2];
#else
  int64_t hidden_size = y_grad.dims()[1];
#endif
  int64_t num_experts = gate_logits_grad.dims()[1];

  apply_moe_dispatch_bwd<T>(y_grad.data<T>(),
                            combine_weights.data<float>(),
                            scatter_index.data<int>(),
                            combine_weights_grad.data<float>(),
                            expert_id.data<int>(),
                            const_cast<float*>(gate_logits_grad.data<float>()),
                            const_cast<T*>(x_grad.data<T>()),
                            num_rows,
                            k,
                            hidden_size,
                            num_experts,
                            capacity,
                            use_all2all_permute,
                            world_size,
                            num_local_experts,
                            dev_ctx.stream());
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
  auto y_grad_dims = y_grad.dims();
  auto scatter_index_dims = scatter_index.dims();

#ifdef MOE_OPS_AUTO
  // y_grad shape is [num_experts, capacity, h]
  int64_t num_experts = y_grad_dims[0];
  int64_t hidden_size = y_grad_dims[2];
#else
  int64_t num_experts = y_grad_dims[0] / capacity;
  int64_t hidden_size = y_grad_dims[1];
#endif
  int64_t num_rows = scatter_index_dims[1];

  const std::vector<int32_t> axis = {1, 0};

  DenseTensor t_scatter_index;
  phi::Transpose<int, Context>(dev_ctx, scatter_index, axis, &t_scatter_index);
  DenseTensor t_scatter_index_;
  phi::ContiguousKernel<int, Context>(
      dev_ctx, t_scatter_index, &t_scatter_index_);
  const DenseTensor t_scatter_index__ = t_scatter_index_;

  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<float>(gate_logits_grad);

  moe_dispatch_bwd<T, Context>(dev_ctx,
                               combine_weights,
                               t_scatter_index__,
                               expert_id,
                               y_grad,
                               combine_weights_grad,
                               *x_grad,
                               *gate_logits_grad,
                               capacity);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
