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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {

template <typename T>
__global__ void cal_aux_loss_grad_kernel(const T* out_grad,
                                         const T* gate_prob,
                                         const int64_t row_gate_prob,
                                         const int64_t col_gate_prob,
                                         const T* seqlen_float,
                                         const T* ce,
                                         const int64_t num_experts,
                                         const bool use_group,
                                         const int64_t moe_k,
                                         T* gate_prob_grad) {
  T ce_val = ce[threadIdx.x];
  T l_aux_grad = *out_grad;
  if (use_group) {
    l_aux_grad = l_aux_grad / static_cast<T>(moe_k);
  }
  l_aux_grad *= static_cast<T>(num_experts);

  gate_prob_grad[blockIdx.x * col_gate_prob + threadIdx.x] =
      (ce_val * l_aux_grad) / (*seqlen_float);
}

template <typename T>
void cal_aux_loss_grad(const T* out_grad,
                       const T* gate_prob,
                       const int64_t row_gate_prob, /*seq_len*/
                       const int64_t col_gate_prob, /*expert_num*/
                       const T* seqlen_float,
                       const T* ce,
                       const int64_t num_experts,
                       const bool use_group,
                       const int64_t moe_k,
                       T* gate_prob_grad,
                       cudaStream_t stream) {
  cal_aux_loss_grad_kernel<T>
      <<<row_gate_prob, col_gate_prob, 0, stream>>>(out_grad,
                                                    gate_prob,
                                                    row_gate_prob,
                                                    col_gate_prob,
                                                    seqlen_float,
                                                    ce,
                                                    num_experts,
                                                    use_group,
                                                    moe_k,
                                                    gate_prob_grad);
}

template <typename T, typename Context>
void CalAuxLossGradKernel(const Context& dev_ctx,
                          const DenseTensor& gate_prob,
                          const DenseTensor& seqlen_float,
                          const DenseTensor& ce,
                          const DenseTensor& l_aux_loss_grad,
                          const int64_t num_experts,
                          const bool use_group,
                          const int64_t moe_k,
                          DenseTensor* gate_prob_grad) {
  auto gate_prob_dims = gate_prob.dims();

  const T* l_aux_loss_grad_data = l_aux_loss_grad.data<T>();
  const T* gate_prob_data = gate_prob.data<T>();
  const T* seqlen_float_data = seqlen_float.data<T>();
  const T* ce_data = ce.data<T>();

  int64_t row_gate_prob = gate_prob_dims[0];
  int64_t col_gate_prob = gate_prob_dims[1];

  T* gate_prob_grad_data = dev_ctx.template Alloc<T>(gate_prob_grad);

  cal_aux_loss_grad<T>(l_aux_loss_grad_data,
                       gate_prob_data,
                       row_gate_prob,
                       col_gate_prob,
                       seqlen_float_data,
                       ce_data,
                       num_experts,
                       use_group,
                       moe_k,
                       gate_prob_grad_data,
                       dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(cal_aux_loss_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CalAuxLossGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
