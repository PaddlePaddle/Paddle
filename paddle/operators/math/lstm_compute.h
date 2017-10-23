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

#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

typedef enum {
  HL_ACTIVATION_SIGMOID = 0,
  HL_ACTIVATION_RELU = 1,
  HL_ACTIVATION_TANH = 2,
  HL_ACTIVATION_LINEAR = 3,
  HL_ACTIVATION_END
} activation_mode_t;

template <class T>
struct LstmMetaValue {
  T *gateValue;
  T *prevStateValue;
  T *stateValue;
  T *stateActiveValue;
  T *outputValue;
  T *checkIg;
  T *checkFg;
  T *checkOg;
};

template <class T>
struct LstmMetaGrad {
  T *gateGrad;
  T *prevStateGrad;
  T *stateGrad;
  T *stateActiveGrad;
  T *outputGrad;
  T *checkIgGrad;
  T *checkFgGrad;
  T *checkOgGrad;
};

inline activation_mode_t ActiveType(const std::string &type) {
  if (type == "sigmoid") {
    return HL_ACTIVATION_SIGMOID;
  } else if (type == "relu") {
    return HL_ACTIVATION_RELU;
  } else if (type == "tanh") {
    return HL_ACTIVATION_TANH;
  } else if (type == "linear" || type == "identity" || type == "") {
    return HL_ACTIVATION_LINEAR;
  } else {
    PADDLE_THROW("Do not support activation type.");
  }
}

template <typename Place, typename T>
class LstmUnitFunctor {
 public:
  static void compute(const platform::DeviceContext &context,
                      LstmMetaValue<T> value, int frame_size, int batch_size,
                      const std::string &gate_act, const std::string &cell_act,
                      const std::string &cand_act);
};

template <typename Place, typename T>
class LstmUnitGradFunctor {
 public:
  static void compute(const platform::DeviceContext &context,
                      LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                      int frame_size, int batch_size,
                      const std::string &gate_act, const std::string &cell_act,
                      const std::string &cand_act);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
