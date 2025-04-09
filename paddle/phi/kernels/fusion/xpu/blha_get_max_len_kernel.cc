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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#include "xpu/xdnn.h"

namespace phi {
namespace fusion {
template <typename Context>
void GetMaxLenTensor(const Context& dev_ctx,
                     const phi::DenseTensor& seq_lens_tensor,
                     const phi::DenseTensor& batch_size,
                     DenseTensor* out) {
  phi::DenseTensor max_len_tensor;
  max_len_tensor.Resize({{1}});
  auto* max_len_tensor_data = dev_ctx.template Alloc<int>(
      &max_len_tensor, max_len_tensor.numel() * sizeof(int));
  const int bsz = batch_size.dims()[0];
  int r = baidu::xpu::api::reduce_max<int>(dev_ctx.x_context(),
                                           seq_lens_tensor.data<int>(),
                                           max_len_tensor_data,
                                           {bsz},
                                           {0});
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::Fatal("baidu::xpu::api::reduce_max failed."));
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
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::BlhaGetMaxLenKernel,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(1).SetBackend(phi::Backend::CPU);
}
