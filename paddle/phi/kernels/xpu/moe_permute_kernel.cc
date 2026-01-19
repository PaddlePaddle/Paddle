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
#define MAX_NUM_EXPERTS 80
#endif

template <typename T, typename Context>
void dispatch_tokens_unzip_stable(const Context &dev_ctx,
                                  const DenseTensor &X,
                                  const DenseTensor &expert_routemap_topk,
                                  const DenseTensor &expert_prob_topk,
                                  const optional<DenseTensor> &XScale,
                                  const DenseTensor &expert_offsets,
                                  DenseTensor *X_unzipped,
                                  DenseTensor *zipped_expertwise_rowmap,
                                  DenseTensor *token_prob_unzipped,
                                  DenseTensor *XScale_unzipped,
                                  const int total_zipped_tokens_num,
                                  const int token_length,
                                  const int total_tokens_after_broadcast,
                                  const int topk,
                                  const int num_experts,
                                  const int scale_length,
                                  const bool do_gather) {
#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()
#define GET_XPU_DATA(tensor, type, xpu_type) \
  reinterpret_cast<const xpu_type *>(tensor.data<type>())
#define GET_PTR_XPU_DATA(tensor, type, xpu_type) \
  reinterpret_cast<xpu_type *>(tensor->data<type>())

#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, DO_GATHER)            \
  using XPU_TOKEN_T = typename XPUTypeTrait<TOKEN_T>::Type;                    \
  using XPU_PROB_T = typename XPUTypeTrait<PROB_T>::Type;                      \
  using XPU_INT_T = typename XPUTypeTrait<INT_T>::Type;                        \
                                                                               \
  int r = xpu::moe_permute<XPU_TOKEN_T, XPU_INT_T, XPU_PROB_T>(                \
      dev_ctx.x_context(),                                                     \
      reinterpret_cast<const XPU_TOKEN_T *>(                                   \
          X.data<TOKEN_T>()), /* hidden_states */                              \
      (XScale ? XScale.get_ptr()->data<float>() : nullptr), /* scale */        \
      reinterpret_cast<const XPU_INT_T *>(                                     \
          expert_routemap_topk.data<INT_T>()), /* expert_routemap_topk */      \
      reinterpret_cast<const XPU_PROB_T *>(                                    \
          expert_prob_topk.data<PROB_T>()), /* expert_prob_topk */             \
      reinterpret_cast<const XPU_INT_T *>(                                     \
          expert_offsets.data<int>()), /* expert_base_offset */                \
      reinterpret_cast<XPU_TOKEN_T *>(                                         \
          X_unzipped->data<TOKEN_T>()), /* hidden_states_unzipped */           \
      reinterpret_cast<XPU_INT_T *>(                                           \
          zipped_expertwise_rowmap                                             \
              ->data<INT_T>()), /* zipped_expertwise_rowmap */                 \
      reinterpret_cast<XPU_PROB_T *>(                                          \
          token_prob_unzipped->data<PROB_T>()),      /* token_prob_unzipped */ \
      XScale_unzipped->data<float>(),                /* scale_unzipped */      \
      static_cast<int64_t>(total_zipped_tokens_num), /* sequence_length */     \
      static_cast<int64_t>(token_length),            /* hidden_size */         \
      static_cast<int64_t>(                                                    \
          total_tokens_after_broadcast), /* total_tokens_after_broadcast */    \
      static_cast<int64_t>(topk),        /* topk */                            \
      static_cast<int64_t>(num_experts), /* num_experts */                     \
      128,                               /* num_scale */                       \
      DO_GATHER                          /* do_gather */                       \
  );                                                                           \
                                                                               \
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "moe_permute");

#define HANDLE_GATHER_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE) \
  if (do_gather) {                                            \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, true)    \
  } else {                                                    \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE, false)   \
  }

// HANDLE_GATHER_CASE(phi::float8_e4m3fn, PROB_T, INT_T, true)
#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                    \
  if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                    \
    HANDLE_GATHER_CASE(phi::bfloat16, PROB_T, INT_T, false) \
  } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {        \
    PADDLE_THROW(common::errors::Unimplemented(             \
        "moe_permute input only support bfloat16"));        \
  }

#define HANDLE_PROB_TYPE(INT_T)                                \
  if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {        \
    PADDLE_THROW(common::errors::Unimplemented(                \
        "moe_permute expert_prob_topk only support float32")); \
  } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) {  \
    HANDLE_TOKEN_TYPE(float, INT_T)                            \
  }

  if (DTYPE_CASE(zipped_expertwise_rowmap->dtype(), INT32)) {
    HANDLE_PROB_TYPE(int)
  }

#undef DTYPE_CASE
#undef GET_DATA
#undef GET_XPU_DATA
#undef GET_PTR_XPU_DATA
#undef DISPATCH_CASE
#undef HANDLE_EXPERT_CASE
#undef HANDLE_TOKEN_TYPE
#undef HANDLE_PROB_TYPE
}

template <typename T, typename Context>
void MoePermuteKernel(const Context &dev_ctx,
                      const DenseTensor &X,  // hidden_states
                      const optional<DenseTensor> &XScale,
                      const DenseTensor &expert_routemap_topk,
                      const DenseTensor &expert_prob_topk,
                      const int num_experts,
                      const std::vector<int> &tokens_per_expert,
                      const int padding_multiplex,
                      const bool do_gather,
                      const bool using_ue8m0_scale,
                      DenseTensor *X_unzipped,
                      DenseTensor *zipped_expertwise_rowmap,
                      DenseTensor *token_prob_unzipped,
                      DenseTensor *XScale_unzipped) {
  const int64_t rows = X.dims()[0];
  const int64_t cols = X.dims()[1];
  PADDLE_ENFORCE_LE(
      rows,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[0] should be less than "
                                      "INT_MAX, received X.dims()[0]: (%ld)",
                                      rows));
  PADDLE_ENFORCE_LE(
      cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[1] should be less than "
                                      "INT_MAX, received X.dims()[1]: (%ld)",
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
  const int64_t quanted_cols = (XScale) ? XScale.get_ptr()->dims()[1] : 0;
  PADDLE_ENFORCE_LE(
      quanted_cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("quanted_cols should be less than "
                                      "INT_MAX, received quanted_cols: (%ld)",
                                      quanted_cols));

  // Expert base offset initialization, tensor numeric range [0, max_token_num]
  int expert_offset[MAX_NUM_EXPERTS];
  int tokens_cumulated = 0;
  for (int i = 0; i < MAX_NUM_EXPERTS; i++) {
    if (i < num_experts) {
      expert_offset[i] = tokens_cumulated;
      tokens_cumulated +=
          ((tokens_per_expert[i] + padding_multiplex - 1) / padding_multiplex) *
          padding_multiplex;
    } else {
      expert_offset[i] = 0;
    }
  }
  DenseTensor expert_offset_tensor;
  expert_offset_tensor.Resize({MAX_NUM_EXPERTS});
  dev_ctx.template Alloc<int>(&expert_offset_tensor);
  PADDLE_ENFORCE_XPU_SUCCESS(
      cudaMemcpyAsync(expert_offset_tensor.data<int>(),
                      expert_offset,
                      sizeof(int) * MAX_NUM_EXPERTS,
                      cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(dev_ctx.stream())));
  // ------------------- resource allocate -------------------------
  const int output_rows = tokens_cumulated;
  const int64_t topk = expert_routemap_topk.dims()[1];
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  token_prob_unzipped->Resize({output_rows});
  if (do_gather) {  // no gather, no resize.
    X_unzipped->Resize({output_rows, cols});
    if (XScale) {
      const int quanted_cols = XScale.get_ptr()->dims()[1];
      XScale_unzipped->Resize({output_rows, quanted_cols});
    }
  }
  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<float>(XScale_unzipped);
  dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
  dev_ctx.template Alloc<float>(token_prob_unzipped);
  auto X_unzipped_ptr = reinterpret_cast<void *>(X_unzipped->data<T>());
  auto token_prob_unzipped_ptr =
      reinterpret_cast<void *>(token_prob_unzipped->data<float>());
  auto XScale_unzipped_ptr =
      reinterpret_cast<void *>(XScale_unzipped->data<float>());

  // -------- Memset all padding area to zero, with regard to do_gather
  auto memset_invalid_rows =
      [&](void *ptr, int64_t element_size, int64_t stride) {
        for (int i = 0; i < num_experts; i++) {
          int64_t next_expert_offset =
              i < num_experts - 1 ? expert_offset[i + 1] : output_rows;
          int64_t invalid_rows =
              next_expert_offset - expert_offset[i] - tokens_per_expert[i];
          int64_t cur_expert_end = expert_offset[i] + tokens_per_expert[i];

          PADDLE_ENFORCE_XPU_SUCCESS(cudaMemsetAsync(
              ptr + cur_expert_end * stride * element_size,
              0,
              element_size * invalid_rows * stride,
              reinterpret_cast<cudaStream_t>(dev_ctx.stream())));
        }
      };
  if (do_gather) {  // no gather, no memset
    memset_invalid_rows(X_unzipped_ptr, sizeof(T), cols);
    if (XScale) {
      memset_invalid_rows(XScale_unzipped_ptr, sizeof(float), quanted_cols);
    }
  }
  // Probs will be memset to zero whatsoever
  memset_invalid_rows(token_prob_unzipped_ptr, sizeof(float), 1);

  // Handle 0-size input
  if (X.numel() == 0) return;

  // -------- Initialize semaphore for cumsum ---------------
  dispatch_tokens_unzip_stable<T, Context>(dev_ctx,
                                           X,
                                           expert_routemap_topk,
                                           expert_prob_topk,
                                           XScale,
                                           expert_offset_tensor,
                                           X_unzipped,
                                           zipped_expertwise_rowmap,
                                           token_prob_unzipped,
                                           XScale_unzipped,
                                           static_cast<int>(rows),
                                           static_cast<int>(cols),
                                           static_cast<int>(output_rows),
                                           static_cast<int>(topk),
                                           num_experts,
                                           static_cast<int>(quanted_cols),
                                           do_gather);
}
#undef MAX_NUM_EXPERTS
}  // namespace phi

PD_REGISTER_KERNEL(moe_permute,
                   XPU,
                   ALL_LAYOUT,
                   phi::MoePermuteKernel,
                   //  phi::float8_e4m3fn,
                   phi::bfloat16) {}
