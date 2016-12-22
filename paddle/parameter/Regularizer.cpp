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

#include "Regularizer.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

namespace paddle {

Regularizer* Regularizer::get(const std::vector<ParameterType>& types,
                              const ParameterConfig& paraConfig) {
  bool useLearningRateVec =
      std::find(types.begin(), types.end(), PARAMETER_LEARNING_RATE) !=
      types.end();
  if (paraConfig.decay_rate_l1() > 0.0f &&
      paraConfig.decay_rate() > 0.0f) {  // use L1 and L2
    if (useLearningRateVec) {
      static L1L2LrRegularizer regularizer_;
      return &regularizer_;
    }
    static L1L2Regularizer regularizer_;
    return &regularizer_;
  }
  if (paraConfig.decay_rate_l1() > 0.0f) {  // use L1 only
    if (useLearningRateVec) {
      static L1LrRegularizer regularizer_;
      return &regularizer_;
    }
    static L1Regularizer regularizer_;
    return &regularizer_;
  }
  if (paraConfig.decay_rate() > 0.0f) {  // use L2 only
    if (useLearningRateVec) {
      static L2LrRegularizer regularizer_;
      return &regularizer_;
    }
    static L2Regularizer regularizer_;
    return &regularizer_;
  }
  return nullptr;
}

}  // namespace paddle
