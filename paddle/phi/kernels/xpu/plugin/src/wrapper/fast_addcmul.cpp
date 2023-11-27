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
template <typename T>
__attribute__((global)) void fast_addcmul(const T* x,
                                          const T* y,
                                          T* z,
                                          int64_t len);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int xpu2_wrapper(
    Context* ctx, const T* w, const T* x, const T* y, T* z, int64_t len) {
  if (x == w) {
    xpu2::plugin::fast_addcmul<T>
        <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, z, len);
  } else {
    return addcmul(ctx, w, x, y, z, 1.0f, len);
  }
  return SUCCESS;
}

template <typename T>
int fast_addcmul(
    Context* ctx, const T* w, const T* x, const T* y, T* z, int64_t len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "fast_mul_add", T);
  WRAPPER_DUMP_PARAM4(ctx, w, x, y, z);
  WRAPPER_DUMP_PARAM2(ctx, len, ctx->_l3_mgr.get_size());
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_4PTRS(ctx, T, len, w, x, y, z);
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx, w, x, y, z, len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_addcmul(
    Context*, const float*, const float*, const float*, float*, int64_t);
template int fast_addcmul(Context*,
                          const float16*,
                          const float16*,
                          const float16*,
                          float16*,
                          int64_t);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
