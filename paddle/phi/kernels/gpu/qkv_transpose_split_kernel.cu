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

#include "paddle/phi/kernels/qkv_transpose_split_kernel.h"

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "cub/cub.cuh"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
namespace {
constexpr int VEC_16B = 16;

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(
    T *q_buf,
    T *k_buf,
    T *v_buf,
    const T *qkv,
    const int *padding_offset,
    const int *seq_lens,
    const int32_t elem_cnt,
    const int batch_size,
    const int max_len_this_time,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int kv_head_num,
    const int size_per_head) {
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = hidden_size + kv_head_num * size_per_head + kv_head_num * size_per_head;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    const int32_t qkv_id = bias_idx < hidden_size ? 0 : (bias_idx -  hidden_size) / ( kv_head_num * size_per_head) + 1;
    const int32_t head_id = qkv_id == 0 ? bias_idx / size_per_head : (bias_idx -  hidden_size) / size_per_head % kv_head_num;
    const int32_t size_id = bias_idx % size_per_head;

    if (qkv_id == 0) {
      phi::Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else if (qkv_id == 1) {
      phi::Store<T, VecSize>(
          src_vec,
          &k_buf[target_batch_id * kv_head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else {
      phi::Store<T, VecSize>(
          src_vec,
          &v_buf[target_batch_id * kv_head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    }
  }
}

} // namespace

namespace phi {

template <typename T, typename Context>
void QkvTransposeSplitKernel(const Context& dev_ctx,
                        const DenseTensor& qkv,
                        const DenseTensor& padding_offset,
                        const DenseTensor& seq_lens,
                        const DenseTensor& input_ids,
                        int num_head,
                        int head_size,
                        DenseTensor* q_out,
                        DenseTensor* k_out,
                        DenseTensor* v_out) {
    using DataType_ = typename PDDataTypeTraits<T>::DataType;
    auto cu_stream = dev_ctx.stream();

    const auto* qkv_input = &qkv;
    const auto& qkv_dims = qkv_input->dims();

    const auto* seq_input = &seq_lens;
    const auto& seq_dims = seq_input->dims();

    const int token_num = qkv_dims[0];
    const int bsz = seq_lens.dims()[0];
    const int max_seq_len = input_ids.dims()[1];

    int64_t fused_hidden_size = qkv_dims[1];
    int kv_num_head = (fused_hidden_size - num_head * head_size) / head_size / 2;

    q_out->Resize({bsz, num_head, max_seq_len, head_size});
    k_out->Resize({bsz, kv_num_head, max_seq_len, head_size});
    v_out->Resize({bsz, kv_num_head, max_seq_len, head_size});


    T* q_out_ptr = dev_ctx.template Alloc<T>(q_out);
    T* k_out_ptr = dev_ctx.template Alloc<T>(k_out);
    T* v_out_ptr = dev_ctx.template Alloc<T>(v_out);

    if (!k_out_ptr || !k_out_ptr || !v_out_ptr) return;

    funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, q_out, static_cast<T>(0));
    set_zero(dev_ctx, k_out, static_cast<T>(0));
    set_zero(dev_ctx, v_out, static_cast<T>(0));

    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    const int elem_cnt = qkv_dims[0] * qkv_dims[1];
    const int pack_num = elem_cnt / PackSize;
    const int blocksize = 128;
    const int grid_size = (pack_num + blocksize - 1) / blocksize;

    fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, cu_stream>>>(
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        qkv.data<T>(),
        padding_offset.data<int>(),
        seq_lens.data<int>(),
        elem_cnt,
        bsz,
        max_seq_len,
        max_seq_len,
        token_num,
        num_head,
        kv_num_head,
        head_size);
    return;
}

}  // namespace phi

PD_REGISTER_KERNEL(qkv_transpose_split,
                   GPU,
                   ALL_LAYOUT,
                   phi::QkvTransposeSplitKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}