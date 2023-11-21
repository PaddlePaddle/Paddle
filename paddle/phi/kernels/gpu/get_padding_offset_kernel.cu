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
__global__ void RemovePaddingKernel(int64_t *output_data,
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

__global__ void GetCumOffsetKernel(int *token_num,
                                   int *enc_token_num,
                                   int *dec_token_num,
                                   int *cum_offsets,
                                   const int *sequence_lengths,
                                   const int *sequence_lengths_encoder,
                                   const int *sequence_lengths_decoder,
                                   const int batch_size,
                                   const int max_seq_len) {
  // get padding offset of each batch
  int total_seq_len = 0;
  int enc_total_seq_len = 0;
  int dec_total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;

  for (int i = 0; i < batch_size; i++) {
    cum_offsets[i] = cum_offset;
    int seq_len = sequence_lengths[i];
    int seq_len_enc = sequence_lengths_encoder[i];
    int seq_len_dec = sequence_lengths_decoder[i];

    cum_offset += max_seq_len - seq_len;

    total_seq_len += seq_len;
    enc_total_seq_len += seq_len_enc;
    dec_total_seq_len += seq_len_dec;
  }
  token_num[0] = total_seq_len;
  enc_token_num[0] = enc_total_seq_len;
  dec_token_num[0] = dec_total_seq_len;
}

}  // namespace

namespace phi {

__global__ void GetPaddingOffsetKernel(int32_t *padding_offset,
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

template <typename T, typename Context>
void GetPaddingOffset(const Context &dev_ctx,
                      const DenseTensor &input_ids,
                      const DenseTensor &cum_offsets,
                      const DenseTensor &token_num,
                      const DenseTensor &seq_len,
                      DenseTensor *x_remove_padding,
                      DenseTensor *cum_offsets_out,
                      DenseTensor *padding_offset) {
  dev_ctx.template Alloc<int64_t>(x_remove_padding);
  dev_ctx.template Alloc<int>(cum_offsets_out);
  dev_ctx.template Alloc<int32_t>(padding_offset);

  auto cu_stream = dev_ctx.stream();
  const int bsz = input_ids.dims()[0];
  const int seq_length = input_ids.dims()[1];

  auto cum_offset_size = sizeof(int) * cum_offsets.numel();
  phi::memory_utils::Copy(cum_offsets.place(),
                          cum_offsets_out->data<int>(),
                          cum_offsets.place(),
                          cum_offsets.data<int>(),
                          cum_offset_size,
                          cu_stream);
  // 从 GPU 复制 token_num 到 CPU 内存
  DenseTensor cpu_token_num;
  cpu_token_num.Resize(token_num.dims());
  auto token_num_size = sizeof(int64_t) * token_num.numel();
  phi::memory_utils::Copy(token_num.place(),
                          cpu_token_num.data<int64_t>(),
                          paddle::CPUPlace(),
                          token_num.data<int64_t>(),
                          token_num_size,
                          cu_stream);
  const int token_num_data = cpu_token_num.data<int64_t>()[0];
  int blockSize = min((token_num_data + 32 - 1) / 32 * 32, 128);
  GetPaddingOffsetKernel<<<bsz, 128, 0, cu_stream>>>(
      padding_offset->data<int>(),
      cum_offsets_out->data<int>(),
      cum_offsets.data<int>(),
      seq_len.data<int>(),
      seq_length);
  RemovePaddingKernel<<<bsz, blockSize, 0, cu_stream>>>(
      x_remove_padding->data<int64_t>(),
      input_ids.data<int64_t>(),
      seq_len.data<int>(),
      cum_offsets_out->data<int>(),
      seq_length);
}
}  // namespace phi

PD_REGISTER_KERNEL(get_padding_offset,
                   GPU,
                   ALL_LAYOUT,
                   phi::GetPaddingOffset,
                   int64_t,
                   int32_t) {}
