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

#pragma once
#include "xpu/xdnn.h"

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

DLL_EXPORT int add2(Context* ctx, const float* x, float* y, int len);
template <typename T>
DLL_EXPORT int fast_where(Context* ctx,
                          const bool* condition,
                          const T* x,
                          const T* y,
                          T* out,
                          int64_t len);
template <typename T, typename TID>

DLL_EXPORT int fast_gather_nd(Context* ctx,
                              const T* x,
                              const TID* index,
                              T* y,
                              const VectorParam<int64_t>& xshape,
                              const std::vector<int64_t>& index_shape);
template <typename T, typename TID>
static inline int fast_gather_nd(Context* ctx,
                                 const T* x,
                                 const TID* index,
                                 T* y,
                                 const VectorParam<int>& xshape,
                                 const std::vector<int>& index_shape) {
  auto deleter = [](int64_t* ptr) { delete[] ptr; };
  std::shared_ptr<int64_t> xshape_i64(new int64_t[xshape.len], deleter);
  return fast_gather_nd(
      ctx,
      x,
      index,
      y,
      vpi32_to_vpi64(xshape, xshape_i64.get()),
      std::vector<int64_t>(index_shape.begin(), index_shape.end()));
}

template <typename T, typename TID>
DLL_EXPORT int take_along_axis(Context* ctx,
                               const T* x,
                               const TID* index,
                               T* y,
                               const std::vector<int64_t>& xshape,
                               const std::vector<int64_t>& idxshape,
                               int64_t axis);

template <typename T>
DLL_EXPORT int fast_layer_norm(Context* ctx,
                               const T* x,
                               T* y,
                               int64_t m,
                               int64_t n,
                               float eps,
                               const float* scale,
                               const float* bias);

template <typename T>
DLL_EXPORT int fast_reduce_sum(Context* ctx,
                               const T* x,
                               T* y,
                               const std::vector<int>& xshape,
                               const std::vector<int>& rdims);

template <typename T>
DLL_EXPORT int fast_reduce_mean(Context* ctx,
                                const T* x,
                                T* y,
                                const std::vector<int>& xshape,
                                const std::vector<int>& rdims);

template <typename T>
DLL_EXPORT int fast_reduce_max(Context* ctx,
                               const T* x,
                               T* y,
                               const std::vector<int>& xshape,
                               const std::vector<int>& rdims);

template <typename T>
DLL_EXPORT int fast_reduce_min(Context* ctx,
                               const T* x,
                               T* y,
                               const std::vector<int>& xshape,
                               const std::vector<int>& rdims);

template <typename T, typename TID>
DLL_EXPORT int fast_embedding(Context* ctx,
                              const T* x,
                              const TID* indices,
                              T* y,
                              int64_t xm,
                              int64_t n,
                              int64_t ym,
                              int64_t padding_idx,
                              TID start_index = 0);
template <typename T>
DLL_EXPORT int fast_addcmul(
    Context* ctx, const T* w, const T* x, const T* y, T* z, int64_t len);

template <typename T>
DLL_EXPORT int bn_act_fusion_infer(Context* ctx,
                                   const T* x,
                                   T* y,
                                   int64_t n,
                                   int64_t c,
                                   int64_t h,
                                   int64_t w,
                                   float eps,
                                   const float* scale,
                                   const float* bias,
                                   const float* global_mean,
                                   const float* global_var,
                                   bool is_nchw,
                                   int act_type);

template <typename T>
DLL_EXPORT int layer_norm_act_fusion(Context* ctx,
                                     const T* x,
                                     T* y,
                                     int64_t m,
                                     int64_t n,
                                     float eps,
                                     const float* scale,
                                     const float* bias,
                                     const Activation_t& act);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
