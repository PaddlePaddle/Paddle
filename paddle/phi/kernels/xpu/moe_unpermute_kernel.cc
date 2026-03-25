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
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/utils/optional.h"

namespace phi {
#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 64
#endif

template <typename T, typename Context>
void dispatch_tokens_zip(const Context &dev_ctx,
                         const DenseTensor &unzipped_tokens,
                         const DenseTensor &zipped_expertwise_rowmap,
                         const DenseTensor &expert_routemap_topk,
                         const DenseTensor &unzipped_token_probs,
                         DenseTensor *zipped_tokens,
                         DenseTensor *zipped_probs_topk,
                         const int total_zipped_tokens_num,
                         const int num_experts,
                         const int token_length,
                         const int topk,
                         const bool MP) {
  using XPU_BF16 = typename XPUTypeTrait<phi::bfloat16>::Type;
  // Map data types to C++ types
  if (unzipped_token_probs.dtype() == paddle::DataType::FLOAT32) {
    int r = xpu::moe_unpermute(
        dev_ctx.x_context(),
        reinterpret_cast<const XPU_BF16 *>(
            unzipped_tokens.data<phi::bfloat16>()),
        reinterpret_cast<const int *>(zipped_expertwise_rowmap.data<int>()),
        reinterpret_cast<const int *>(expert_routemap_topk.data<int>()),
        reinterpret_cast<const float *>(unzipped_token_probs.data<float>()),
        reinterpret_cast<XPU_BF16 *>(zipped_tokens->data<phi::bfloat16>()),
        zipped_probs_topk->data<float>(),
        total_zipped_tokens_num,
        num_experts,
        token_length,
        topk,
        MP,
        unzipped_tokens.dims()[0]);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "moe_unpermute");
  }
}

template <typename T, typename Context>
void MoeUnpermuteKernel(const Context &dev_ctx,
                        const DenseTensor &unzipped_tokens,
                        const DenseTensor &zipped_expertwise_rowmap,
                        const DenseTensor &expert_routemap_topk,
                        const DenseTensor &unzipped_token_probs,
                        const int total_zipped_tokens_num,
                        const int num_experts,
                        const bool MP,
                        const bool using_weighted_combine,
                        DenseTensor *zipped_tokens,
                        DenseTensor *zipped_probs_topk) {
  PADDLE_ENFORCE_EQ(
      using_weighted_combine,
      false,
      common::errors::Unimplemented("moe_unpermute on XPU does not support "
                                    "using_weighted_combine=true yet."));
  const int64_t cols = unzipped_tokens.dims()[1];
  PADDLE_ENFORCE_LE(cols,
                    std::numeric_limits<int32_t>::max(),
                    common::errors::InvalidArgument(
                        "unzipped_tokens.dims()[1] should be less than "
                        "INT_MAX, received unzipped_tokens.dims()[1]: (%ld)",
                        cols));
  PADDLE_ENFORCE_LE(
      num_experts,
      MAX_NUM_EXPERTS,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          MAX_NUM_EXPERTS,
          num_experts));
  const int64_t topk = expert_routemap_topk.dims()[1];
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  dev_ctx.template Alloc<T>(zipped_tokens);
  dev_ctx.template Alloc<float>(zipped_probs_topk);
  if (unzipped_tokens.numel() == 0) return;  // 0-size tensor
  void *zipped_probs_topk_ptr =
      reinterpret_cast<void *>(zipped_probs_topk->data<float>());
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaMemsetAsync(zipped_probs_topk_ptr,
                      0,
                      sizeof(float) * int64_t(total_zipped_tokens_num) * topk,
                      reinterpret_cast<cudaStream_t>(dev_ctx.stream())));
  dispatch_tokens_zip<T, Context>(dev_ctx,
                                  unzipped_tokens,
                                  zipped_expertwise_rowmap,
                                  expert_routemap_topk,
                                  unzipped_token_probs,
                                  zipped_tokens,
                                  zipped_probs_topk,
                                  total_zipped_tokens_num,
                                  num_experts,
                                  static_cast<int>(cols),
                                  static_cast<int>(topk),
                                  MP);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    moe_unpermute, XPU, ALL_LAYOUT, phi::MoeUnpermuteKernel, phi::bfloat16) {}
