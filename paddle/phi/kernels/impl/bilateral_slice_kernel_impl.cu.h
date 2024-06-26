// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {

struct GridSizes {
  int64_t h;
  int64_t w;
  int64_t bs;
  int64_t coeffs_chans;
  int64_t gd;
  int64_t gh;
  int64_t gw;
  int64_t input_chans;
};

template <typename T>
inline __device__ T DiffAbs(T x) {
  T eps = 1e-8;
  return sqrt(x * x + eps);
}

template <typename T>
inline __device__ T DdiffAbs(T x) {
  T eps = 1e-8;
  return x / sqrt(x * x + eps);
}

template <typename T>
inline __device__ T WeightZ(T x) {
  T abx = DiffAbs(x);
  return max(1.0f - abx, 0.0f);
}

template <typename T>
inline __device__ T DweightZ(T x) {
  T abx = DiffAbs(x);
  if (abx > 1.0f) {
    return 0.0f;
  } else {
    return DdiffAbs(x);
  }
}

}  // namespace phi
