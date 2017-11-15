/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

#include "paddle/operators/math/lstm_compute.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

// TODO(guosheng): refine code style in gru_compute
template <typename T>
struct hl_gru_value {
  T *gateWeight;
  T *stateWeight;
  T *gateValue;
  T *resetOutputValue;
  T *outputValue;
  T *prevOutValue;
};

template <typename T>
struct hl_gru_grad {
  T *gateWeightGrad;
  T *stateWeightGrad;
  T *gateGrad;
  T *resetOutputGrad;
  T *outputGrad;
  T *prevOutGrad;
};

template <typename Place, typename T>
struct GRUUnitFunctor {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, int frameSize, int batchSize,
                      activation_mode_t active_node,
                      activation_mode_t active_gate);
};

template <typename Place, typename T>
struct GRUUnitGradFunctor {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, hl_gru_grad<T> grad, int frameSize,
                      int batchSize, activation_mode_t active_node,
                      activation_mode_t active_gate);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
