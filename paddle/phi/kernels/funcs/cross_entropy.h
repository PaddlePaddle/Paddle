/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <limits>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
namespace funcs {

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    PADDLE_ENFORCE(std::is_floating_point<T>::value,
                   "TolerableValue should be float in cross_entropy.");
    const T kApproInf = 1e20;

    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

// NOTE(dzh): float16 value clip behave different.
// 1. Our ValueClipping has a  hardcore threshold 1e20
// for float number. 1e20 will resulting in overflow in float16.
// 2. float16 should expose the real number overflow to python.
// because mixed-training depends the inf/nan value to determine
// if the scale value will be adjusted.
// Also. In standard implementation of cross entropy, other
// framework not has the ValueClipping.
template <>
struct TolerableValue<phi::dtype::float16> {
  HOSTDEVICE phi::dtype::float16 operator()(
      const phi::dtype::float16& x) const {
    if (phi::dtype::isfinite(x)) {
      return x;
    } else if (x > static_cast<phi::dtype::float16>(0)) {
      return std::numeric_limits<phi::dtype::float16>::max();
    } else {
      return std::numeric_limits<phi::dtype::float16>::min();
    }
  }
};

template <>
struct TolerableValue<phi::dtype::bfloat16> {
  HOSTDEVICE phi::dtype::bfloat16 operator()(
      const phi::dtype::bfloat16& x) const {
    if (phi::dtype::isfinite(x)) {
      return x;
    } else if (x > static_cast<phi::dtype::bfloat16>(0)) {
      return std::numeric_limits<phi::dtype::bfloat16>::max();
    } else {
      return std::numeric_limits<phi::dtype::bfloat16>::min();
    }
  }
};

template <typename DeviceContext, typename T>
class CrossEntropyFunctor {
 public:
  void operator()(const DeviceContext& context,
                  phi::DenseTensor* out,
                  const phi::DenseTensor* prob,
                  const phi::DenseTensor* labels,
                  const bool softLabel,
                  const int ignore_index,
                  const int axis_dim);
};
}  // namespace funcs
}  // namespace phi
