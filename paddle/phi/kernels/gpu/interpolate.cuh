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
#pragma once

namespace phi {

template <typename MT>
__device__ __forceinline__ void ComputeWeightsSpan(const int i,
                                                   const int input_size,
                                                   const MT scale,
                                                   const MT support,
                                                   int* xmin,
                                                   int* xsize,
                                                   MT* center) {
  *center = scale * (i + static_cast<MT>(0.5));
  *xmin = max(static_cast<int>(*center - support + static_cast<MT>(0.5)), 0);
  *xsize = min(static_cast<int>(*center + support + static_cast<MT>(0.5)),
               input_size) -
           *xmin;
}

// Compute single weight on-the-fly without storing all weights
// This is used when shared memory is insufficient for large ratio values
template <typename MT, typename InterpFilter>
__device__ __forceinline__ MT ComputeSingleWeight(const MT scale,
                                                  const InterpFilter& filter,
                                                  const MT xmin_m_center,
                                                  int idx) {
  MT invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  return filter((idx + xmin_m_center + static_cast<MT>(0.5)) * invscale);
}

// Compute weight normalization factor (sum of all weights)
template <typename MT, typename InterpFilter>
__device__ __forceinline__ MT ComputeWeightSum(const MT scale,
                                               const InterpFilter& filter,
                                               const MT xmin_m_center,
                                               int xsize) {
  MT invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  MT total_w = 0.0;
  for (int j = 0; j < xsize; j++) {
    total_w += filter((j + xmin_m_center + static_cast<MT>(0.5)) * invscale);
  }
  return total_w;
}

// Compute single normalized weight for backward pass on-the-fly
template <typename MT, typename InterpFilter>
__device__ __forceinline__ MT
ComputeSingleWeightBwNormalized(const MT scale,
                                const InterpFilter& filter,
                                const MT xmin_m_center,
                                int idx,
                                const MT total_w) {
  MT invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  MT w = filter((idx + xmin_m_center + static_cast<MT>(0.5)) * invscale);
  if (total_w != 0.0) {
    w /= total_w;
  }
  return w;
}

}  // namespace phi
