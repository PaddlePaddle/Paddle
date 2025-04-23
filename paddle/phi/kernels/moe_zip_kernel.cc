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
#include "paddle/phi/kernels/moe_zip_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MoeZipKernel(const Context& dev_ctx,
                  const DenseTensor& unzipped_tokens,
                  const DenseTensor& zipped_expertwise_rowmap,
                  const DenseTensor& expert_routemap_topk,
                  const DenseTensor& unzipped_token_probs,
                  DenseTensor* zipped_tokens,
                  DenseTensor* zipped_prob_topk) {
  PADDLE_THROW(
      common::errors::Unimplemented("MoeZipKernel is not implemented."));
}
}  // namespace phi

PD_REGISTER_KERNEL(moe_zip,
                   CPU,
                   ALL_LAYOUT,
                   phi::MoeZipKernel,
                   int,
                   float,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(moe_zip,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeZipKernel,
                   int,
                   float,
                   phi::dtype::bfloat16) {}
#endif
