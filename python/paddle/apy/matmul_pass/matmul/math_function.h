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

#include <cuda.h>
#include <cuda_fp16.h>

namespace ap {

template <typename T>
__forceinline__ __host__ __device__ T ComputePow(T base, T exponent) {
  T res = (exponent == static_cast<T>(2))
              ? (base * base)
              : ((exponent == static_cast<T>(3)) ? (base * base * base)
                                                 : (powf(base, exponent)));
  return res;
}

}  // namespace ap
