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

#include "paddle/utils/Logging.h"

#include <fstream>

#include "AverageOptimizer.h"
#include "FirstOrderOptimizer.h"
#include "OptimizerFunctions.h"
#include "OptimizerWithRegularizer.h"
#include "ParameterOptimizer.h"
#include "hl_gpu.h"

namespace paddle {

ParameterOptimizer* ParameterOptimizer::create(
    const OptimizationConfig& optConfig, bool inPserver) {
  if (inPserver && optConfig.num_batches_per_send_parameter() > 1) {
    return new AddOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "momentum") {
    return new SgdOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "torch_momentum") {
    return new SgdOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "adagrad") {
    return new AdagradParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "adadelta") {
    return new AdaDeltaParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "rmsprop") {
    return new RMSPropParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "decayed_adagrad") {
    return new DecayedAdagradParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "adam") {
    return new AdamParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "adamax") {
    return new AdamaxParameterOptimizer(optConfig);
  }
  if (optConfig.learning_method() == "sparse_momentum") {
    return new SparseMomentumParameterOptimizer(optConfig);
  }
  return nullptr;
}

}  // namespace paddle
