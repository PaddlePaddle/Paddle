// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"

namespace phi {

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;

template <typename T, typename U>
inline HOSTDEVICE auto copysign_func(const T& a, const U& b) {
  return copysign(static_cast<double>(a), static_cast<double>(b));
}

inline HOSTDEVICE float16 copysign_func(const float16& x, const float16& y) {
  return float16((x.x & 0x7fff) | (y.x & 8000));
}

inline HOSTDEVICE bfloat16 copysign_func(const bfloat16& x, const bfloat16& y) {
  return bfloat16((x.x & 0x7fff) | (y.x & 8000));
}

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out);
}  // namespace  phi
