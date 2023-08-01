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

namespace xpu2 {
namespace plugin {
template <typename T>
__attribute__((global)) void fast_where(
    const int8_t* condition, const T* x, const T* y, T* z, int64_t len);
}
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx,
                       const bool* condition,
                       const T* x,
                       const T* y,
                       T* z,
                       int64_t len) {
  for (int64_t i = 0; i < len; i++) {
    z[i] = condition[i] ? x[i] : y[i];
  }
  return SUCCESS;
}

template <>
int cpu_wrapper<float16>(Context* ctx,
                         const bool* condition,
                         const float16* x,
                         const float16* y,
                         float16* z,
                         int64_t len) {
  std::vector<float> x_fp32(len);
  std::vector<float> y_fp32(len);
  std::vector<float> z_fp32(len);
  int ret = cast<float16, float>(ctx, x, x_fp32.data(), len);
  ret = cast<float16, float>(ctx, y, y_fp32.data(), len);
  ret = cpu_wrapper<float>(
      ctx, condition, x_fp32.data(), y_fp32.data(), z_fp32.data(), len);
  ret = cast<float, float16>(ctx, z_fp32.data(), z, len);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return ret;
}

template <typename T>
static int xpu2_wrapper(Context* ctx,
                        const bool* condition,
                        const T* x,
                        const T* y,
                        T* z,
                        int64_t len) {
  xpu2::plugin::fast_where<T><<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      reinterpret_cast<const int8_t*>(condition), x, y, z, len);
  return SUCCESS;
}

template <typename T>
int fast_where(Context* ctx,
               const bool* condition,
               const T* x,
               const T* y,
               T* z,
               int64_t len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "fast_where", float);
  WRAPPER_DUMP_PARAM5(ctx, condition, x, y, z, len);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, len, 0);
  WRAPPER_CHECK_2PTRS(ctx, T, len, x, y);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx, condition, x, y, z, len);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx, condition, x, y, z, len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_where(Context*,
                        const bool* condition,
                        const float*,
                        const float*,
                        float*,
                        int64_t);
template int fast_where(Context*,
                        const bool* condition,
                        const float16*,
                        const float16*,
                        float16*,
                        int64_t);
template int fast_where(Context*,
                        const bool* condition,
                        const int16_t*,
                        const int16_t*,
                        int16_t*,
                        int64_t);
template int fast_where(Context*,
                        const bool* condition,
                        const int32_t*,
                        const int32_t*,
                        int32_t*,
                        int64_t);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
