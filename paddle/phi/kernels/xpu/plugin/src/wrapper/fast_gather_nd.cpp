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
template <typename TID>
__attribute__((global)) void fast_gather1d(const int8_t* x,
                                           const TID* index,
                                           int64_t count,
                                           int64_t x_dim0,
                                           int64_t x_stride0,
                                           int8_t* y);
template <typename TID>
__attribute__((global)) void fast_gather2d(const int8_t* x,
                                           const TID* index,
                                           int64_t count,
                                           int64_t x_dim0,
                                           int64_t x_dim1,
                                           int64_t x_stride0,
                                           int64_t x_stride1,
                                           int8_t* y);
template <typename TID>
__attribute__((global)) void fast_gather3d(const int8_t* x,
                                           const TID* index,
                                           int64_t count,
                                           int64_t x_dim0,
                                           int64_t x_dim1,
                                           int64_t x_dim2,
                                           int64_t x_stride0,
                                           int64_t x_stride1,
                                           int64_t x_stride2,
                                           int8_t* y);
template <typename TID>
__attribute__((global)) void fast_gather4d(const int8_t* x,
                                           const TID* index,
                                           int64_t count,
                                           int64_t x_dim0,
                                           int64_t x_dim1,
                                           int64_t x_dim2,
                                           int64_t x_dim3,
                                           int64_t x_stride0,
                                           int64_t x_stride1,
                                           int64_t x_stride2,
                                           int64_t x_stride3,
                                           int8_t* y);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T, typename TID>
static int cpu_wrapper(Context* ctx,
                       const T* x,
                       const TID* index,
                       T* y,
                       const VectorParam<int64_t>& x_shape,
                       const std::vector<int64_t>& index_shape) {
  int64_t x_shape_size = x_shape.len;
  int64_t index_shape_size = index_shape.size();
  int64_t gather_time = 1;
  for (int64_t i = 0; i < index_shape_size - 1; i++) {
    gather_time *= index_shape[i];
  }
  int64_t end_size = index_shape.back();
  int64_t gather_size = 1;
  for (int64_t i = end_size; i < x_shape_size; i++) {
    gather_size *= x_shape.cpu[i];
  }
  const int64_t gather_bytes = gather_size * sizeof(T);
  for (int64_t i = 0; i < gather_time; i++) {
    int64_t x_index = 0;
    int64_t step = 1;
    for (int64_t j = end_size - 1; j >= 0; j--) {
      x_index += (index[i * end_size + j] * step);
      step *= x_shape.cpu[j];
    }
    memcpy(y, x + x_index * gather_size, gather_bytes);
    y += gather_size;
  }
  return api::SUCCESS;
}

template <typename T, typename TID>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        const TID* index,
                        T* y,
                        const VectorParam<int64_t>& x_shape,
                        const std::vector<int64_t>& index_shape) {
  using XPU_TID = typename XPUIndexType<TID>::type;
  int64_t x_shape_size = x_shape.len;
  int64_t index_shape_size = index_shape.size();
  int64_t end_size = index_shape.back();
  int64_t gather_time = 1;
  for (int64_t i = 0; i < index_shape_size - 1; i++) {
    gather_time *= index_shape[i];
  }
  std::vector<int64_t> gather_strides(end_size);
  gather_strides[end_size - 1] = sizeof(T);
  for (int64_t i = end_size; i < x_shape_size; i++) {
    gather_strides[end_size - 1] *= x_shape.cpu[i];
  }
  for (int64_t i = end_size - 2; i >= 0; i--) {
    gather_strides[i] = gather_strides[i + 1] * x_shape.cpu[i + 1];
  }
  auto casted_x = static_cast<const int8_t*>(static_cast<const void*>(x));
  auto casted_index =
      static_cast<const XPU_TID*>(static_cast<const void*>(index));
  auto casted_y = static_cast<int8_t*>(static_cast<void*>(y));
  switch (end_size) {
    case 1:
      xpu2::plugin::fast_gather1d<XPU_TID>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(casted_x,
                                                     casted_index,
                                                     gather_time,
                                                     x_shape.cpu[0],
                                                     gather_strides[0],
                                                     casted_y);
      return api::SUCCESS;
    case 2:
      xpu2::plugin::fast_gather2d<XPU_TID>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(casted_x,
                                                     casted_index,
                                                     gather_time,
                                                     x_shape.cpu[0],
                                                     x_shape.cpu[1],
                                                     gather_strides[0],
                                                     gather_strides[1],
                                                     casted_y);
      return api::SUCCESS;
    case 3:
      xpu2::plugin::fast_gather3d<XPU_TID>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(casted_x,
                                                     casted_index,
                                                     gather_time,
                                                     x_shape.cpu[0],
                                                     x_shape.cpu[1],
                                                     x_shape.cpu[2],
                                                     gather_strides[0],
                                                     gather_strides[1],
                                                     gather_strides[2],
                                                     casted_y);
      return api::SUCCESS;
    case 4:
      xpu2::plugin::fast_gather4d<XPU_TID>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(casted_x,
                                                     casted_index,
                                                     gather_time,
                                                     x_shape.cpu[0],
                                                     x_shape.cpu[1],
                                                     x_shape.cpu[2],
                                                     x_shape.cpu[3],
                                                     gather_strides[0],
                                                     gather_strides[1],
                                                     gather_strides[2],
                                                     gather_strides[3],
                                                     casted_y);
      return api::SUCCESS;
    default:
      break;
  }
  return gather_nd(ctx, x, index, y, x_shape, index_shape);
}

template <typename T, typename TID>
int fast_gather_nd(Context* ctx,
                   const T* x,
                   const TID* index,
                   T* y,
                   const VectorParam<int64_t>& x_shape,
                   const std::vector<int64_t>& index_shape) {
  WRAPPER_CHECK_CTX(ctx);
  if (std::is_same<T, bfloat16>::value) {
    WRAPPER_UNIMPLEMENTED(ctx);
  }
  WRAPPER_DUMP_FUNCTION_T2(ctx, "fast_gather_nd", T, TID);
  WRAPPER_DUMP_PARAM6(
      ctx, x, index, y, x_shape, index_shape, ctx->_l3_mgr.get_size());
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, x_shape.len, 0);
  WRAPPER_ASSERT_LE(ctx, x_shape.len, 32);
  WRAPPER_ASSERT_GT(ctx, index_shape.size(), 0);
  int64_t x_len = 1;
  for (int64_t i = 0; i < x_shape.len; i++) {
    x_len *= x_shape.cpu[i];
  }
  WRAPPER_CHECK_PTR(ctx, T, x_len, x);
  int64_t index_len = -1;
  WRAPPER_CHECK_SHAPE(ctx, &index_len, index_shape);
  WRAPPER_CHECK_PTR(ctx, TID, index_len, index);
  // index.shape[-1] <= x.rank
  WRAPPER_ASSERT_LE(ctx, index_shape.back(), x_shape.len);
  std::vector<int64_t> y_shape;
  for (int64_t i = 0; i < index_shape.size() - 1; i++) {
    y_shape.push_back(index_shape[i]);
  }
  for (int64_t i = index_shape.back(); i < x_shape.len; i++) {
    y_shape.push_back(x_shape.cpu[i]);
  }
  int64_t y_len = -1;
  WRAPPER_CHECK_SHAPE(ctx, &y_len, y_shape);
  WRAPPER_CHECK_PTR(ctx, T, y_len, y);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T, TID>(ctx, x, index, y, x_shape, index_shape);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T, TID>(ctx, x, index, y, x_shape, index_shape);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_gather_nd(Context*,
                            const float*,
                            const int*,
                            float*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const int*,
                            const int*,
                            int*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const int64_t*,
                            const int*,
                            int64_t*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const float16*,
                            const int*,
                            float16*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const float*,
                            const int64_t*,
                            float*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const int*,
                            const int64_t*,
                            int*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const int64_t*,
                            const int64_t*,
                            int64_t*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const float16*,
                            const int64_t*,
                            float16*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const bfloat16*,
                            const int*,
                            bfloat16*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);
template int fast_gather_nd(Context*,
                            const bfloat16*,
                            const int64_t*,
                            bfloat16*,
                            const VectorParam<int64_t>&,
                            const std::vector<int64_t>&);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
