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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES  // use M_2_SQRTPI on Windows
#endif
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void ErfinvKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<T>(out);
  auto eigen_in = EigenVector<T>::Flatten(x);
  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto& place = *ctx.eigen_device();
  constexpr T half = static_cast<T>(0.5);
  constexpr T half_sqrt = static_cast<T>(M_SQRT1_2);
  eigen_out.device(place) = (eigen_in * half + half).ndtri() * half_sqrt;
}

}  // namespace phi
