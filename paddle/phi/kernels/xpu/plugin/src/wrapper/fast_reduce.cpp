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
__attribute__((global)) void fast_reduce_sum_tiny(const T* x,
                                                  T* y,
                                                  int m,
                                                  int t);
template <typename T>
__attribute__((global)) void fast_reduce_mean_tiny(const T* x,
                                                   T* y,
                                                   int m,
                                                   int t);
template <typename T>
__attribute__((global)) void fast_reduce_max_tiny(const T* x,
                                                  T* y,
                                                  int m,
                                                  int t);
template <typename T>
__attribute__((global)) void fast_reduce_min_tiny(const T* x,
                                                  T* y,
                                                  int m,
                                                  int t);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx,
                       const T* x,
                       T* y,
                       const std::vector<int>& xshape,
                       int op_type) {
  std::vector<int> yshape = xshape;
  yshape[xshape.size() - 1] = 1;
  int xlen = vector_prod(xshape);
  int ylen = vector_prod(yshape);
  int num = xlen / ylen;
  if (op_type == 0 || op_type == 1) {
    for (int i = 0; i < ylen; i++) {
      T sum = 0;
      for (int j = 0; j < num; j++) {
        sum += x[i * num + j];
      }
      if (op_type == 0) {
        y[i] = sum;
      } else {
        y[i] = sum / num;
      }
    }
  } else if (op_type == 2) {
    for (int i = 0; i < ylen; i++) {
      T max_val = x[i * num];
      for (int j = 1; j < num; j++) {
        max_val = std::max<T>(max_val, x[i * num + j]);
      }
      y[i] = max_val;
    }
  } else if (op_type == 3) {
    for (int i = 0; i < ylen; i++) {
      T min_val = x[i * num];
      for (int j = 1; j < num; j++) {
        min_val = std::min<T>(min_val, x[i * num + j]);
      }
      y[i] = min_val;
    }
  }

  return SUCCESS;
}

template <>
int cpu_wrapper<float16>(Context* ctx,
                         const float16* x,
                         float16* y,
                         const std::vector<int>& xshape,
                         int op_type) {
  std::vector<int> yshape = xshape;
  yshape[xshape.size() - 1] = 1;
  int xlen = vector_prod(xshape);
  int ylen = vector_prod(yshape);
  std::vector<float> x_fp32(xlen);
  std::vector<float> y_fp32(ylen);
  int ret = cast<float16, float>(ctx, x, x_fp32.data(), xlen);
  ret = cpu_wrapper<float>(ctx, x_fp32.data(), y_fp32.data(), xshape, op_type);
  ret = cast<float, float16>(ctx, y_fp32.data(), y, ylen);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return ret;
}

template <typename T>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        T* y,
                        const std::vector<int>& xshape,
                        int op_type) {
  std::vector<int> rdims = {static_cast<int>(xshape.size() - 1)};
  switch (op_type) {
    case 0:
      return reduce_sum<T>(ctx, x, y, xshape, rdims);
    case 2:
      return reduce_max<T>(ctx, x, y, xshape, rdims);
    case 3:
      return reduce_min<T>(ctx, x, y, xshape, rdims);
    default:
      return NOT_IMPLEMENT;
  }
  return SUCCESS;
}

template <>
int xpu2_wrapper<int8_t>(Context* ctx,
                         const int8_t* x,
                         int8_t* y,
                         const std::vector<int>& xshape,
                         int op_type) {
  std::vector<int> rdims = {static_cast<int>(xshape.size() - 1)};
  if (op_type == 0) {
    return reduce_sum<int8_t>(ctx, x, y, xshape, rdims);
  } else {
    return NOT_IMPLEMENT;
  }
  return SUCCESS;
}

template <>
int xpu2_wrapper<float>(Context* ctx,
                        const float* x,
                        float* y,
                        const std::vector<int>& xshape,
                        int op_type) {
  int t = xshape[xshape.size() - 1];
  int xlen = vector_prod(xshape);
  int m = xlen / t;
  switch (op_type) {
    case 0:
      xpu2::plugin::fast_reduce_sum_tiny<float>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 1:
      xpu2::plugin::fast_reduce_mean_tiny<float>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 2:
      xpu2::plugin::fast_reduce_max_tiny<float>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 3:
      xpu2::plugin::fast_reduce_min_tiny<float>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    default:
      return NOT_IMPLEMENT;
  }
  return SUCCESS;
}

template <>
int xpu2_wrapper<float16>(Context* ctx,
                          const float16* x,
                          float16* y,
                          const std::vector<int>& xshape,
                          int op_type) {
  int t = xshape[xshape.size() - 1];
  int xlen = vector_prod(xshape);
  int m = xlen / t;
  switch (op_type) {
    case 0:
      xpu2::plugin::fast_reduce_sum_tiny<float16>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 1:
      xpu2::plugin::fast_reduce_mean_tiny<float16>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 2:
      xpu2::plugin::fast_reduce_max_tiny<float16>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    case 3:
      xpu2::plugin::fast_reduce_min_tiny<float16>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, m, t);
      break;
    default:
      return NOT_IMPLEMENT;
  }
  return SUCCESS;
}

template <typename T>
int fast_reduce_tiny(Context* ctx,
                     const T* x,
                     T* y,
                     const std::vector<int>& xshape,
                     int op_type) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "fast_reduce_tiny", T);
  WRAPPER_DUMP_PARAM4(ctx, x, y, xshape, op_type);
  WRAPPER_DUMP(ctx);
  std::vector<int> yshape = xshape;
  yshape[xshape.size() - 1] = 1;
  int64_t lenx = -1;
  int64_t leny = -1;
  WRAPPER_CHECK_SHAPE(ctx, &lenx, xshape);
  WRAPPER_CHECK_SHAPE(ctx, &leny, yshape);
  WRAPPER_CHECK_PTR(ctx, T, lenx, x);
  WRAPPER_CHECK_PTR(ctx, T, leny, y);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx, x, y, xshape, op_type);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx, x, y, xshape, op_type);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_reduce_tiny(
    Context*, const float*, float*, const std::vector<int>&, int);
template int fast_reduce_tiny(
    Context*, const float16*, float16*, const std::vector<int>&, int);
template int fast_reduce_tiny(
    Context*, const int*, int*, const std::vector<int>&, int);
template int fast_reduce_tiny(
    Context*, const int64_t*, int64_t*, const std::vector<int>&, int);
template int fast_reduce_tiny(
    Context*, const int8_t*, int8_t*, const std::vector<int>&, int);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
