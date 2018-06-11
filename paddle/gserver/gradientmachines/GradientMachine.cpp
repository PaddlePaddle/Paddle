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

#include "GradientMachine.h"

#include <fstream>
#include "paddle/utils/Logging.h"

#include "NeuralNetwork.h"
#include "hl_gpu.h"

#ifndef PADDLE_MOBILE_INFERENCE
#include "GradientMachineMode.h"
#include "MultiGradientMachine.h"
#include "MultiNetwork.h"
#include "ParallelNeuralNetwork.h"
#endif

namespace paddle {

GradientMachine* GradientMachine::create(
    const ModelConfig& config,
    int mode,
    const std::vector<ParameterType>& parameterTypes) {
#ifndef PADDLE_MOBILE_INFERENCE
  if (auto gm = IGradientMachineMode::tryCreateGradientMachine(mode, config)) {
    return gm;
  }
  if (FLAGS_trainer_count > 1) {
    return new MultiGradientMachine(config, FLAGS_use_gpu);
  }
#endif
  if (FLAGS_trainer_count == 1) {  // single
#ifndef PADDLE_MOBILE_INFERENCE
    NeuralNetwork* nn;
    if (config.type() == "multi_nn") {
      /* multi submodel calculate, thread(s) will be initialized inside */
      nn = new MultiNetwork("root");
    } else if (FLAGS_parallel_nn) {
      /* multi threads calculate */
      nn = new ParallelNeuralNetwork();
    } else {
      /* single thread calculate */
      nn = NeuralNetwork::create(config);
    }
#else
    NeuralNetwork* nn = NeuralNetwork::create(config);
#endif
    ParamInitCallback testParamInitCb = [](int paramId, Parameter* para) {
      para->enableType(PARAMETER_VALUE);
    };
    nn->init(
        config, mode == kTesting ? testParamInitCb : nullptr, parameterTypes);
    return nn;
  }
  LOG(FATAL) << "Unknown model type: " << config.type();
  return nullptr;
}

void GradientMachine::saveParameters(const std::string& dir) const {
  LOG(INFO) << "Saving parameters to " << dir;

  for (auto& para : parameters_) {
    std::string filename = dir + "/" + para->getName();
    if (para->isFullSize()) {
      para->save(filename);
    }
  }
}

void GradientMachine::loadParameters(const std::string& dir) {
  LOG(INFO) << "Loading parameters from " << dir;

  for (auto& para : parameters_) {
    std::string filename = dir + "/" + para->getName();
    if (para->isFullSize()) {
      para->load(filename);
    }
  }
}

void GradientMachine::randParameters() {
  LOG(INFO) << "Initing parameters..";

  for (auto& para : parameters_) {
    if (para->isFullSize()) {
      para->randomize();
    }
  }
  LOG(INFO) << "Init parameters done.";
}

}  // namespace paddle
