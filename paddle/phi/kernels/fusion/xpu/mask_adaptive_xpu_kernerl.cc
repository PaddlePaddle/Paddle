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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MaskAdaptiveXPUKernel(const Context& ctx,
                           const DenseTensor& mask,
                           DenseTensor* length,
                           DenseTensor* seq_lod,
                           DenseTensor* pad_seq_len) {
  const auto& mask_dims = mask.dims();
  auto batch_size = mask_dims[0];
  auto pad_seq_len_size = mask_dims[1];
  std::vector<int> cpu_seq_lod{0};
  std::vector<int64_t> cpu_seq_lens;
  auto mask_ptr = mask.data<T>();
  for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int cur_batch_seq_len = 0;
    for (auto seq_idx = 0; seq_idx < pad_seq_len_size; seq_idx++) {
      if (mask_ptr[batch_idx * pad_seq_len_size + seq_idx] >
          static_cast<T>(1e-7)) {
        cur_batch_seq_len += 1;
      } else {
        break;
      }
    }
    PADDLE_ENFORCE_GT(
        cur_batch_seq_len,
        0,
        common::errors::InvalidArgument("cur_batch_seq_len [%d] is less than 0",
                                        cur_batch_seq_len));
    cpu_seq_lod.push_back(cpu_seq_lod.back() + cur_batch_seq_len);
    cpu_seq_lens.push_back(cur_batch_seq_len);
  }
  auto* seq_lod_ptr = ctx.template HostAlloc<int>(seq_lod);
  memcpy(seq_lod_ptr, cpu_seq_lod.data(), cpu_seq_lod.size() * sizeof(int));
  auto* seq_lens_ptr = ctx.template HostAlloc<int64_t>(length);
  memcpy(
      seq_lens_ptr, cpu_seq_lens.data(), cpu_seq_lens.size() * sizeof(int64_t));
  auto* pad_seq_len_ptr = ctx.template HostAlloc<int>(pad_seq_len);
  pad_seq_len_ptr[0] = pad_seq_len_size;
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(mask_adaptive_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MaskAdaptiveXPUKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(1).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(2).SetBackend(phi::Backend::CPU);
}
