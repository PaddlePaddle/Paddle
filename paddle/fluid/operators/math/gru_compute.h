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

#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct GRUMetaValue {
  T *gate_weight;
  T *state_weight;
  T *gate_value;
  T *reset_output_value;
  T *output_value;
  T *prev_out_value;
};

template <typename T>
struct GRUMetaGrad {
  T *gate_weight_grad;
  T *state_weight_grad;
  T *gate_grad;
  T *reset_output_grad;
  T *output_grad;
  T *prev_out_grad;
};

template <typename DeviceContext, typename T>
struct GRUUnitFunctor {
  static void compute(const DeviceContext &context, GRUMetaValue<T> value,
                      int frame_size, int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate,
                      bool origin_mode);
};

template <typename DeviceContext, typename T>
struct GRUUnitGradFunctor {
  static void compute(const DeviceContext &context, GRUMetaValue<T> value,
                      GRUMetaGrad<T> grad, int frame_size, int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate,
                      bool origin_mode);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
