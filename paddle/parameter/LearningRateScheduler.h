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

#include "TrainerConfig.pb.h"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {
// NOLINTNEXTLINES_4
#define REGISTER_LEARNING_RATE_SCHEDULER(__type_name, __class_name) \
  static InitFunction __reg_type_##__type_name([]() {               \
    LearningRateScheduler::registrar_.registerClass<__class_name>(  \
        #__type_name);                                              \
  })

class LearningRateScheduler {
public:
  static LearningRateScheduler* create(const OptimizationConfig& config);
  virtual ~LearningRateScheduler() {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) = 0;

  static ClassRegistrar<LearningRateScheduler, OptimizationConfig> registrar_;
};

}  // namespace paddle
