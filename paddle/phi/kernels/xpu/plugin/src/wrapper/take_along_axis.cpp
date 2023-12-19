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
template <typename T, typename TID>
__attribute__((global)) void take_along_axis(const T* x,
                                             const TID* indices,
                                             T* y,
                                             int64_t batch,
                                             int64_t xlen,
                                             int64_t ylen);
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
                       const std::vector<int64_t> xshape,
                       const std::vector<int64_t>& idxshape,
                       int64_t axis) {
  int64_t ylen = vector_prod(idxshape);
  for (int64_t i = 0; i < ylen; i++) {
    std::vector<int64_t> sp_x_id = id_to_split_id(idxshape, i);
    sp_x_id[axis] = index[i];

    // -xshape[axis] <= index value < xshape[axis]
    WRAPPER_ASSERT_LT(ctx, sp_x_id[axis], xshape[axis]);
    WRAPPER_ASSERT_GE(ctx, sp_x_id[axis], -xshape[axis]);

    if (sp_x_id[axis] < 0) {
      sp_x_id[axis] += xshape[axis];
    }

    int64_t xid = split_id_to_id(xshape, sp_x_id);
    y[i] = x[xid];
  }
  return SUCCESS;
}

template <typename T, typename TID>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        const TID* index,
                        T* y,
                        const std::vector<int64_t> xshape,
                        const std::vector<int64_t>& idxshape,
                        int64_t axis) {
  int64_t m_idx = 1;
  for (int64_t i = 0; i < axis; i++) {
    m_idx *= idxshape[i];
  }
  int64_t t_x = xshape[axis];
  int64_t t_idx = idxshape[axis];
  int64_t n_idx = vector_prod(idxshape) / m_idx / t_idx;

  if (m_idx < 64 && n_idx == 1) {
    using XPU_TID = typename XPUIndexType<TID>::type;
    const XPU_TID* casted_index =
        static_cast<const XPU_TID*>(static_cast<const void*>(index));

    xpu2::plugin::take_along_axis<T, XPU_TID>
        <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
            x, casted_index, y, m_idx, t_x, t_idx);
  } else {
    return gather_element(ctx, x, index, y, xshape, idxshape, axis);
  }
  return SUCCESS;
}

template <typename T, typename TID>
int take_along_axis(Context* ctx,
                    const T* x,
                    const TID* index,
                    T* y,
                    const std::vector<int64_t>& xshape,
                    const std::vector<int64_t>& idxshape,
                    int64_t axis) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T2(ctx, "take_along_axis", T, TID);
  WRAPPER_DUMP_PARAM6(ctx, x, index, y, xshape, idxshape, axis);
  WRAPPER_DUMP(ctx);
  int64_t xlen = -1;
  WRAPPER_CHECK_SHAPE(ctx, &xlen, xshape);
  WRAPPER_CHECK_PTR(ctx, T, xlen, x);
  int64_t idxlen = -1;
  WRAPPER_CHECK_SHAPE(ctx, &idxlen, idxshape);
  WRAPPER_CHECK_PTR(ctx, TID, idxlen, index);
  WRAPPER_CHECK_PTR(ctx, T, idxlen, y);

  WRAPPER_ASSERT_EQ(
      ctx,
      xshape.size(),
      idxshape.size());  // x and index tensor should have same rank
  int64_t neg_rank = -xshape.size();
  WRAPPER_ASSERT_GE(ctx, axis, neg_rank);
  WRAPPER_ASSERT_LT(ctx, axis, xshape.size());

  axis = (axis < 0) ? (axis + xshape.size()) : axis;
  for (int64_t i = 0; i < xshape.size(); i++) {
    if (i != axis) {
      WRAPPER_ASSERT_EQ(ctx, xshape[i], idxshape[i]);
    }
  }

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T, TID>(ctx, x, index, y, xshape, idxshape, axis);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T, TID>(ctx, x, index, y, xshape, idxshape, axis);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int take_along_axis(Context*,
                             const float*,
                             const int*,
                             float*,
                             const std::vector<int64_t>&,
                             const std::vector<int64_t>&,
                             int64_t);
template int take_along_axis(Context*,
                             const float*,
                             const int64_t*,
                             float*,
                             const std::vector<int64_t>&,
                             const std::vector<int64_t>&,
                             int64_t);
template int take_along_axis(Context*,
                             const float16*,
                             const int*,
                             float16*,
                             const std::vector<int64_t>&,
                             const std::vector<int64_t>&,
                             int64_t);
template int take_along_axis(Context*,
                             const float16*,
                             const int64_t*,
                             float16*,
                             const std::vector<int64_t>&,
                             const std::vector<int64_t>&,
                             int64_t);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
