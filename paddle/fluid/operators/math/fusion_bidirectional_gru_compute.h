/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct FusionGRUMetaValue {
  const T *x;
  const T *wx0;
  const T *wx1;
  const T *wx2;
  const T *wx3;
  const T *bias_x0;
  const T *bias_x1;
  T *mul_o0;
  T *mul_o1;
  const T *bias_h0;
  const T *bias_h1;
  const T *wh0;
  const T *wh1;
  T *hp0;
  T *hp1;
  T *gate0;
  T *gate1;
  T *gru_o0;
  T *gru_o1;
  T *out;
};

template <typename DeviceContext, typename T>
struct FusionBidirectionalGRUFunctor {
  static void compute(const DeviceContext &context, FusionGRUMetaValue<T> v,
                      int m, int n, int k, int q,
                      const detail::ActivationType active_gate,
                      const detail::ActivationType active_node, int reverse);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
