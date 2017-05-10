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

#include "FirstOrderOptimizer.h"

namespace paddle {

/*
 * Factory function creates the corresponding SgdOptimizer
 * according to the configuration in optConfig.
 */
ParameterOptimizer* sgdOptimizerCreate(const OptimizationConfig& optConfig,
                                       const ParameterConfig& paraConfig,
                                       bool isParameterSparse,
                                       bool inPserver);

/*
 * Get the parameter types needed for the specific optimization
 * algorithm specified in optConfig.
 */
std::vector<ParameterType> sgdOptimizerGetTypes(
    const OptimizationConfig& optConfig, bool inPserver);

/*
 * Whether trainer need call apply() in pserver and get result back.
 * currently, only averager depend on this.
 */
bool useApplyInPserver(const OptimizationConfig& optConfig);

}  // namespace paddle
