/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    PADDLE_ASSERT(std::is_floating_point<T>::value);
    const T kApproInf = 1e20;

    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

template <typename DeviceContext, typename T>
class CrossEntropyFunctor {
 public:
  void operator()(const DeviceContext& context, framework::Tensor* out,
                  const framework::Tensor* prob,
                  const framework::Tensor* labels, const bool softLabel);
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
