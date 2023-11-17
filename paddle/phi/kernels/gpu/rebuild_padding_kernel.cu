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
#include "paddle/phi/kernels/rebuild_padding_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace {
template <typename T>
__global__ void RebuildPaddingKernel(T *output_data,
                                     const T *input_data,
                                     const int *padding_offset,
                                     const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dst_seq_id = bid + padding_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[dst_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}
}  // namespace

namespace phi {
constexpr int VEC_16B = 16;

template <typename T, int VecSize>
__global__ void RebuildPaddingKernel(T *output_data,
                                     const T *input_data,
                                     const int *cum_offsets,
                                     const int *seq_lens,
                                     const int max_seq_len,
                                     const int dim_embed,
                                     const int elem_nums) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = seq_lens[bi] - 1;
    const int ori_token_idx = bi * max_seq_len - cum_offsets[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;
    Load<T, VecSize>(&input_data[src_offset], &src_vec);
    Store<T, VecSize>(src_vec, &output_data[i]);
  }
}

template <typename T, typename Context>
void RebuildPaddingKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const DenseTensor &padding_offset,
                          const DenseTensor &seq_lens,
                          const DenseTensor &input_ids,
                          DenseTensor *output) {
  dev_ctx.template Alloc<T>(output);

  auto cu_stream = dev_ctx.stream();

  // 获取输入张量的维度信息
  const int token_num = x.dims()[0];
  const int dim_embed = x.dims()[1];
  const int bsz = seq_lens.dims()[0];

  // 计算用于 CUDA 内核的参数
  constexpr int PackSize = VEC_16B / sizeof(T);
  int elem_nums = output->numel();
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  const int grid_size = (pack_num + blocksize - 1) / blocksize;

  // 调用 CUDA 内核
  RebuildPaddingKernel<T, PackSize><<<grid_size, blocksize, 0, cu_stream>>>(
      reinterpret_cast<T *>(output->data<T>()),
      reinterpret_cast<const T *>(x.data<T>()),
      padding_offset.data<int>(),
      seq_lens.data<int>(),
      input_ids.dims()[1],
      dim_embed,
      elem_nums);
}
}  // namespace phi
PD_REGISTER_KERNEL(rebuild_padding,
                   GPU,
                   ALL_LAYOUT,
                   phi::RebuildPaddingKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
