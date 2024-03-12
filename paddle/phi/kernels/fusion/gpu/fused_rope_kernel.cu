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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
namespace phi {
namespace fusion {

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeKernel(phi::Array<const T*, 3> ins_data,
                                          int64_t batch_size,
                                          int64_t seq_len,
                                          int64_t num_heads,
                                          int64_t head_dim,
                                          phi::Array<T*, 3> outs_data,
                                          int num_inputs,
                                          MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq = index_wc / (num_heads * head_dim);
      MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }

#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter > num_inputs) break;
      const T* input = ins_data[iter] + index;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < kVectorsPerThread; ++nx) {
        int pr_index = nx * 2;
        int ls_index = pr_index + 1;

        MPType p0 = static_cast<MPType>(input[pr_index]);
        MPType p1 = static_cast<MPType>(input[ls_index]);

        result[pr_index] = cos_value[pr_index] * p0;
        result[pr_index] -= sin_value[pr_index] * p1;

        result[ls_index] = sin_value[ls_index] * p0;
        result[ls_index] += cos_value[ls_index] * p1;

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

template <typename T, typename Context>
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const paddle::optional<DenseTensor>& k,
                     const paddle::optional<DenseTensor>& v,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) {
  int64_t numel = q.numel();
  if (numel <= 0) return;
  dev_ctx.template Alloc<T>(out_q);
  // small size for broadcast
  auto batch_size = q.dims()[0];
  auto num_heads = q.dims()[2];
  auto head_dim = q.dims()[3];
  auto seq_len = q.dims()[1];
  PADDLE_ENFORCE_NE(head_dim % 2,
                    1,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));

  constexpr const int vec_size = 2;

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);

  int64_t grid = config.block_per_grid.x;
  int64_t block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();

  phi::Array<T*, 3> outs_data;
  phi::Array<const T*, 3> ins_data;

  ins_data[0] = q.data<T>();
  outs_data[0] = out_q->data<T>();
  int num_inputs = 0;

  if (k.get_ptr()) {
    dev_ctx.template Alloc<T>(out_k);
    ins_data[1] = k->data<T>();
    outs_data[1] = out_k->data<T>();
    num_inputs++;
  }

  if (v.get_ptr()) {
    dev_ctx.template Alloc<T>(out_v);
    ins_data[2] = v->data<T>();
    outs_data[2] = out_v->data<T>();
    num_inputs++;
  }

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType div_c = static_cast<MPType>(1.0f / head_dim);

  VectorizedFusedRopeKernel<T, MPType, vec_size>
      <<<grid, block, 0, stream>>>(ins_data,
                                   batch_size,
                                   seq_len,
                                   num_heads,
                                   head_dim,
                                   outs_data,
                                   num_inputs,
                                   div_c);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_rotary_position_embedding,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedRopeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
