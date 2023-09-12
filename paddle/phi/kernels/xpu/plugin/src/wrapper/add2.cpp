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
__attribute__((global)) void add1(const float* x, float* y, int len);
}
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx, const float* x, float* y, int len) {
  for (int i = 0; i < len; i++) {
    y[i] = x[i] + 2.0f;
  }
  return SUCCESS;
}

static int xpu2_wrapper(Context* ctx, const float* x, float* y, int len) {
  ctx_guard RAII_GUARD(ctx);
  float* tensor_one = RAII_GUARD.alloc<float>(len);
  WRAPPER_ASSERT_WORKSPACE(ctx, tensor_one);
  int ret = constant<float>(ctx, tensor_one, len, 1.0f);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = add<float>(ctx, x, tensor_one, y, len);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  xpu2::plugin::add1<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(y, y, len);
  return api::SUCCESS;
}

int add2(Context* ctx, const float* x, float* y, int len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "add2", float);
  WRAPPER_DUMP_PARAM3(ctx, x, y, len);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, len, 0);
  WRAPPER_CHECK_2PTRS(ctx, float, len, x, y);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx, x, y, len);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper(ctx, x, y, len);
  }
  return NOT_IMPLEMENT;
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
