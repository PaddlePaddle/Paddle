// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/phi/kernels/moe_ops_partial_nosoftmaxtopk_grad_kernel.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/moe_fuse_bwd_op.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T>
void apply_moe_dispatch_bwd(const T* y_grad,
                            const float* combine_weights,  // [s, k]
                            const int* scatter_index,      // [s, k]
                            const float* combine_weights_out_grad,
                            float* combine_weights_in_grad,
                            T* x_grad,
                            int64_t num_rows,
                            int64_t k,
                            int64_t dim,
                            int64_t num_experts,
                            int64_t num_active,
                            cudaStream_t stream) {
  gather_with_mask_launcher<T>(y_grad,
                               scatter_index,
                               combine_weights,
                               x_grad,
                               num_rows,
                               k,
                               dim,
                               num_active,
                               stream);
  auto out_grad_ptr = thrust::device_pointer_cast(combine_weights_out_grad);
  auto in_grad_ptr = thrust::device_pointer_cast(combine_weights_in_grad);
  auto combine_weight_ptr = thrust::device_pointer_cast(combine_weights);
  thrust::transform(thrust::cuda::par.on(stream),
                    out_grad_ptr,
                    out_grad_ptr + num_rows * k,
                    combine_weight_ptr,
                    in_grad_ptr,
                    [] __device__(float g, float w) {
                      return w > static_cast<float>(0) ? g
                                                       : static_cast<float>(0);
                    });
  // topk_grad_with_mask_launcher<float>(combine_weights_grad,
  //                                     expert_id,
  //                                     combine_weights,
  //                                     gate_logtis_grad,
  //                                     num_rows, k, num_experts, stream);
}

template <typename T, typename Context>
void moe_dispatch_bwd(const Context& dev_ctx,
                      const DenseTensor& combine_weights,  // [s, k]
                      const DenseTensor& scatter_index,    // [k, s]
                      const DenseTensor& y_grad,  // [num_experts * capacity, h]
                      const DenseTensor& combine_weights_out_grad,  // [s, k]
                      DenseTensor* x_grad,
                      DenseTensor* combine_weights_in_grad,
                      int64_t num_experts) {
  int64_t num_rows = combine_weights.dims()[0];
  int64_t k = combine_weights.dims()[1];
  int64_t hidden_size = y_grad.dims()[1];
  int64_t num_active = y_grad.dims()[0];

  apply_moe_dispatch_bwd<T>(y_grad.data<T>(),
                            combine_weights.data<float>(),
                            scatter_index.data<int>(),
                            combine_weights_out_grad.data<float>(),
                            combine_weights_in_grad->data<float>(),
                            x_grad->data<T>(),
                            num_rows,
                            k,
                            hidden_size,
                            num_experts,
                            num_active,
                            dev_ctx.stream());
}

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
  DenseTensor t_scatter_index_out;
  phi::ContiguousKernel<int, Context>(
      dev_ctx, t_scatter_index, &t_scatter_index_out);
  t_scatter_index = t_scatter_index_out;
  int64_t num_experts = expert_offset.dims()[0];
  moe_dispatch_bwd<T, Context>(dev_ctx,
                               combine_weights_out,
                               t_scatter_index,
                               y_grad,
                               combine_weights_out_grad,
                               x_grad,
                               combine_weights_grad,
                               num_experts);
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_partial_nosoftmaxtopk_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeGateDispatchPartialNoSoftMaxTopkGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
