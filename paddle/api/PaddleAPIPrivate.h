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
#include <memory>
#include "PaddleAPI.h"
#include "paddle/gserver/evaluators/Evaluator.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/parameter/ParameterUpdaterBase.h"
#include "paddle/trainer/TrainerConfigHelper.h"

struct GradientMachinePrivate {
  std::shared_ptr<paddle::GradientMachine> machine;

  template <typename T>
  inline T& cast(void* ptr) {
    return *(T*)(ptr);
  }
};

struct OptimizationConfigPrivate {
  std::shared_ptr<paddle::TrainerConfigHelper> trainer_config;
  paddle::OptimizationConfig config;

  const paddle::OptimizationConfig& getConfig() {
    if (trainer_config != nullptr) {
      return trainer_config->getOptConfig();
    } else {
      return config;
    }
  }
};

struct TrainerConfigPrivate {
  std::shared_ptr<paddle::TrainerConfigHelper> conf;
  TrainerConfigPrivate() {}
};

struct ModelConfigPrivate {
  std::shared_ptr<paddle::TrainerConfigHelper> conf;
};

struct ArgumentsPrivate {
  std::vector<paddle::Argument> outputs;

  inline paddle::Argument& getArg(size_t idx) throw(RangeError) {
    if (idx < outputs.size()) {
      return outputs[idx];
    } else {
      RangeError e;
      throw e;
    }
  }

  template <typename T>
  std::shared_ptr<T>& cast(void* rawPtr) const {
    return *(std::shared_ptr<T>*)(rawPtr);
  }
};

struct ParameterUpdaterPrivate {
  std::unique_ptr<paddle::ParameterUpdater> updater;
};

struct ParameterPrivate {
  std::shared_ptr<paddle::Parameter> sharedPtr;
  paddle::Parameter* rawPtr;  // rawPtr only used in ParameterUpdater,
                              // in other situation sharedPtr should
                              // contains value.

  ParameterPrivate() : sharedPtr(nullptr), rawPtr(nullptr) {}

  paddle::Parameter* getPtr() {
    if (sharedPtr) {
      return sharedPtr.get();
    } else {
      return rawPtr;
    }
  }
};

struct EvaluatorPrivate {
  paddle::Evaluator* rawPtr;

  EvaluatorPrivate() : rawPtr(nullptr) {}
  ~EvaluatorPrivate() { delete rawPtr; }
};
