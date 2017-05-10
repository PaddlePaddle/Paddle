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

#include "AverageOptimizer.h"
#include "FirstOrderOptimizer.h"
#include "OptimizerWithRegularizer.h"

namespace paddle {

// creator for AverageOptimizer
ParameterOptimizer* sgdOptimizerCreate(const OptimizationConfig& optConfig,
                                       const ParameterConfig& paraConfig,
                                       bool isParameterSparse,
                                       bool inPserver) {
  ParameterOptimizer* optimizer = OptimizerWithRegularizer::create(
      optConfig, paraConfig, isParameterSparse, inPserver);
  return AverageOptimizer::create(
      optConfig, optimizer, isParameterSparse, inPserver /*useParameterApply*/);
}

std::vector<ParameterType> sgdOptimizerGetTypes(
    const OptimizationConfig& optConfig, bool inPserver) {
  std::unique_ptr<ParameterOptimizer> optimizer;
  optimizer.reset(
      AverageOptimizer::create(optConfig,
                               ParameterOptimizer::create(optConfig, inPserver),
                               false /*isParameterSparse*/,
                               inPserver));
  CHECK(optimizer) << "fail to create optimizer: "
                   << optConfig.learning_method();
  return optimizer->getParameterTypes();
}

bool useApplyInPserver(const OptimizationConfig& optConfig) {
  auto types = sgdOptimizerGetTypes(optConfig, true /*inPserver*/);
  return types.end() != std::find(types.begin(), types.end(), PARAMETER_APPLY);
}

}  // namespace paddle
