// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <limits.h>

namespace phi {

enum class Mode {
  bilinear,
  nearest,
};

template <typename T>
__forceinline__ __device__ T SafeDownGradeToIntRange(T x) {
  bool unsafe_cond =
      x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x));
  return unsafe_cond ? static_cast<T>(-100.0) : x;
}

enum class PaddingMode { zeros, border, reflect };

static __forceinline__ __device__ bool InBounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ bool InBounds3D(
    int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

}  // namespace phi
