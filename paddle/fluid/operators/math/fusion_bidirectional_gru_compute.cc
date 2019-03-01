// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/math/fusion_bidirectional_gru_compute.h"
#include <paddle/fluid/platform/device_context.h>
#include <fstream>
#include <sstream>
#include <type_traits>

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct FusionBidirectionalGRUFunctor<platform::CPUDeviceContext, T> {
  static void compute(const platform::CPUDeviceContext &context,
                      FusionGRUMetaValue<T> v, int m, int n, int k, int q,
                      const detail::ActivationType active_gate,
                      const detail::ActivationType active_node, int reverse) {
    PADDLE_THROW("CPU is not support for this kernel now");
  }
};

template struct FusionBidirectionalGRUFunctor<platform::CPUDeviceContext,
                                              float>;
template struct FusionBidirectionalGRUFunctor<platform::CPUDeviceContext,
                                              double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
