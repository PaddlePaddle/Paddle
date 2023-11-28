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

#include "paddle/phi/kernels/get_padding_offset_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace {

__global__ void InvokeRemovePadding(int64_t *output_data,
                                    const int64_t *input_data,
                                    const int *seq_lens,
                                    const int *cum_offsets,
                                    const int sequence_length) {
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;

  for (int i = tid; i < seq_lens[bi]; i += blockDim.x) {
    const int tgt_seq_id = bi * sequence_length - cum_offsets[bi] + i;
    const int src_seq_id = bi * sequence_length + i;
    output_data[tgt_seq_id] = input_data[src_seq_id];
  }
}

__global__ void InvokeGetPaddingOffset(int *padding_offset,
                                       int *cum_offsets_out,
                                       const int *cum_offsets,
                                       const int *seq_lens,
                                       const int max_seq_len) {
  // get padding offset of each batch
  const int bi = blockIdx.x;
  const int ti = threadIdx.x;
  if (ti == 0) {
    cum_offsets_out[bi] = bi == 0 ? 0 : cum_offsets[bi - 1];
  }
  int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
  for (int i = ti; i < seq_lens[bi]; i += blockDim.x) {
    padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
  }
}

}  // namespace

namespace phi {

template <typename T, typename Context>
void GetPaddingOffsetKernel(const Context &dev_ctx,
                            const DenseTensor &input_ids,
                            const DenseTensor &cum_offsets,
                            const DenseTensor &token_num,
                            const DenseTensor &seq_len,
                            DenseTensor *x_remove_padding,
                            DenseTensor *cum_offsets_out,
                            DenseTensor *padding_offset) {
  const int bsz = input_ids.dims()[0];
  const int seq_length = input_ids.dims()[1];

  cum_offsets_out->Resize(phi::make_ddim({bsz}));
  auto cum_offsets_out_ptr = dev_ctx.template Alloc<int32_t>(cum_offsets_out);

  auto cum_offset_size = sizeof(int32_t) * cum_offsets.numel();
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          cum_offsets_out_ptr,
                          cum_offsets.place(),
                          cum_offsets.data<int32_t>(),
                          cum_offset_size,
                          dev_ctx.stream());

  // 从 GPU 复制 token_num 到 CPU 内存
  DenseTensor cpu_token_num(token_num.dtype());
  cpu_token_num.Resize(token_num.dims());
  phi::Copy(dev_ctx, token_num, phi::CPUPlace{}, true, &cpu_token_num);
  const int32_t token_num_data = cpu_token_num.data<int64_t>()[0];

  x_remove_padding->Resize(phi::make_ddim({token_num_data}));
  padding_offset->Resize(phi::make_ddim({token_num_data}));
  auto x_remove_padding_ptr = dev_ctx.template Alloc<int64_t>(x_remove_padding);
  auto padding_offset_ptr = dev_ctx.template Alloc<int32_t>(padding_offset);

  int blockSize = min((token_num_data + 32 - 1) / 32 * 32, 128);
  InvokeGetPaddingOffset<<<bsz, 128, 0, dev_ctx.stream()>>>(
      padding_offset_ptr,
      cum_offsets_out_ptr,
      cum_offsets.data<int32_t>(),
      seq_len.data<int32_t>(),
      seq_length);
  InvokeRemovePadding<<<bsz, blockSize, 0, dev_ctx.stream()>>>(
      x_remove_padding_ptr,
      input_ids.data<int64_t>(),
      seq_len.data<int32_t>(),
      cum_offsets_out_ptr,
      seq_length);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    get_padding_offset, GPU, ALL_LAYOUT, phi::GetPaddingOffsetKernel, int64_t) {
}
