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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"
#include "xpu/refactor/util/vector_util.h"

namespace xpu2 {
namespace plugin {
template <typename emb_idx_type>
__attribute__((global)) void embedding_fwd_kl2_tiny_dict(
    const emb_idx_type* idx,
    const char* dict,
    char* featvec,
    int64_t emb_dim,
    int64_t dict_idx_len,
    int64_t idx_len,
    int64_t padding_idx,
    emb_idx_type start_index);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

// CPU implementation
template <typename T, typename TID>
static int cpu_wrapper(Context* ctx,
                       const T* x,
                       const TID* indices,
                       T* y,
                       int64_t xm,
                       int64_t n,
                       int64_t ym,
                       int64_t padding_idx,
                       TID start_index) {
  for (int64_t i = 0; i < ym; i++) {
    TID real_index = indices[i] - start_index;  // -start_index BEFORE compare
    if (real_index == padding_idx) {
      ::memset(y + i * n, 0, sizeof(T) * n);
    } else {
      if (real_index >= 0 && real_index < xm) {
        std::memcpy(y + i * n, x + real_index * n, sizeof(T) * n);
      } else {
        // set zeros
        for (int64_t k = 0; k < n; ++k) {
          y[i * n + k] = 0;
        }
      }
    }
  }
  return api::SUCCESS;
}

template <typename T, typename TID>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        const TID* indices,
                        T* y,
                        int64_t xm,
                        int64_t n,
                        int64_t ym,
                        int64_t padding_idx,
                        TID start_index) {
  const int TOTAL_LM_SIZE = 6144;  // 6 KB
  int total_emb_dict_size = xm * n * sizeof(T);
  // residual_lm_space = index + result
  int residual_lm_space = TOTAL_LM_SIZE - total_emb_dict_size - 64;
  // The maximum count that can be processed in one iteration.
  int idx_cnt = residual_lm_space / (sizeof(TID) + n * sizeof(T));
  bool plugin_entry_condition = idx_cnt >= 16;
  // This plugin is suitable for scenarios with relatively small dictionary
  // sizes, requiring process greater than 16 index count one iter, in order to
  // load the dictionary into local memory at once, and to leave enough space
  // for the local memory to store the results.
  if (plugin_entry_condition) {
    using XPU_TID = typename XPUIndexType<TID>::type;
    const XPU_TID* casted_indices =
        static_cast<const XPU_TID*>(static_cast<const void*>(indices));
    XPU_TID casted_start_index = static_cast<XPU_TID>(start_index);
    if (ctx->dev().type() == api::kXPU2) {
      xpu2::plugin::embedding_fwd_kl2_tiny_dict<XPU_TID>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
              casted_indices,
              reinterpret_cast<const char*>(x),
              reinterpret_cast<char*>(y),
              n * sizeof(T),
              xm,
              ym,
              padding_idx,
              casted_start_index);
    }
  } else {
    embedding<T, TID>(ctx, x, indices, y, xm, n, ym, padding_idx, start_index);
  }

  return api::SUCCESS;
}

template <typename T, typename TID>
int fast_embedding(Context* ctx,
                   const T* x,
                   const TID* indices,
                   T* y,
                   int64_t xm,
                   int64_t n,
                   int64_t ym,
                   int64_t padding_idx,
                   TID start_index) {
  WRAPPER_CHECK_CTX(ctx);
  if (std::is_same<T, bfloat16>::value) {
    WRAPPER_UNIMPLEMENTED(ctx);
  }
  WRAPPER_DUMP_FUNCTION_T2(ctx, "fast_embedding", T, TID);
  WRAPPER_DUMP_PARAM6(ctx, x, indices, y, xm, n, ym);
  WRAPPER_DUMP_PARAM3(ctx, padding_idx, start_index, ctx->_l3_mgr.get_size());
  WRAPPER_DUMP(ctx);
  int64_t xlen = -1;
  int64_t ylen = -1;
  WRAPPER_CHECK_SHAPE(ctx, &xlen, {xm, n});
  WRAPPER_CHECK_SHAPE(ctx, &ylen, {ym, n});
  WRAPPER_CHECK_PTR(ctx, T, xlen, x);
  WRAPPER_CHECK_PTR(ctx, T, ylen, y);
  WRAPPER_CHECK_PTR(ctx, TID, ym, indices);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(
        ctx, x, indices, y, xm, n, ym, padding_idx, start_index);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T, TID>(
        ctx, x, indices, y, xm, n, ym, padding_idx, start_index);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_embedding(Context*,
                            const float*,
                            const int*,
                            float*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int);
template int fast_embedding(Context*,
                            const float*,
                            const int64_t*,
                            float*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t);
template int fast_embedding(Context*,
                            const float16*,
                            const int*,
                            float16*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int);
template int fast_embedding(Context*,
                            const float16*,
                            const int64_t*,
                            float16*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t);
template int fast_embedding(Context*,
                            const bfloat16*,
                            const int*,
                            bfloat16*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int);
template int fast_embedding(Context*,
                            const bfloat16*,
                            const int64_t*,
                            bfloat16*,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t,
                            int64_t);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
