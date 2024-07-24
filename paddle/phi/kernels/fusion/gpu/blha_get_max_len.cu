// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/fusion/cutlass/variable_length_memory_efficient_attention.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#include "paddle/utils/none.h"

namespace phi {
namespace fusion {

void GetMaxLenTensor(const phi::GPUContext& dev_ctx,
                     const phi::DenseTensor& seq_lens_tensor,
                     const phi::DenseTensor& batch_size,
                     DenseTensor* out) {
  phi::DenseTensor max_len_tensor;
  max_len_tensor.Resize({{1}});
  auto* max_len_tensor_data = dev_ctx.template Alloc<int>(
      &max_len_tensor, max_len_tensor.numel() * sizeof(int));
  const int bsz = batch_size.dims()[0];
  constexpr int blockSize = 128;
  int max_len_cpu = 0;
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, dev_ctx.stream()>>>(
      seq_lens_tensor.data<int>(), max_len_tensor.data<int>(), bsz);
  MemcpyD2HKernel(dev_ctx, max_len_tensor, 0, out);
}

template <typename T, typename Context>
void BlhaGetMaxLenKernel(const Context& dev_ctx,
                         const DenseTensor& seq_lens_encoder,
                         const DenseTensor& seq_lens_decoder,
                         const phi::DenseTensor& batch_size,
                         DenseTensor* max_enc_len_this_time,
                         DenseTensor* max_dec_len_this_time) {
  // decoder
  max_dec_len_this_time->Resize({{1}});
  GetMaxLenTensor(dev_ctx, seq_lens_decoder, batch_size, max_dec_len_this_time);

  // encoder
  max_enc_len_this_time->Resize({{1}});
  GetMaxLenTensor(dev_ctx, seq_lens_encoder, batch_size, max_enc_len_this_time);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(blha_get_max_len,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlhaGetMaxLenKernel,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(1).SetBackend(phi::Backend::CPU);
}
