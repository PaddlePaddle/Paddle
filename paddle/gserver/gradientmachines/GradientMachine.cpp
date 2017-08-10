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

#include "GradientMachine.h"

#include <string.h>
#include <fstream>
#include "paddle/utils/Logging.h"

#include "GradientMachineMode.h"
#include "MultiGradientMachine.h"
#include "MultiNetwork.h"
#include "NeuralNetwork.h"
#include "ParallelNeuralNetwork.h"
#include "hl_gpu.h"

namespace paddle {

GradientMachine* GradientMachine::create(
    const ModelConfig& config,
    int mode,
    const std::vector<ParameterType>& parameterTypes) {
  if (auto gm = IGradientMachineMode::tryCreateGradientMachine(mode, config)) {
    return gm;
  }
  if (FLAGS_trainer_count > 1) {
    return new MultiGradientMachine(config, FLAGS_use_gpu);
  }
  if (FLAGS_trainer_count == 1) {  // single
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

void GradientMachine::loadParameters(const char* buf, uint64_t length) {
  LOG(INFO) << "Loading parameter from pre-load buffer";

  CHECK_NOTNULL(buf);
  CHECK_GE(length, static_cast<uint64_t>(sizeof(uint64_t)));

  uint64_t numFiles = 0;
  memcpy(&numFiles, buf, sizeof(uint64_t));
  uint64_t position = sizeof(uint64_t);
  LOG(INFO) << "numFiles: " << numFiles << ", position: " << position;

  std::map<std::string, char*> offsets;
  std::map<std::string, uint64_t> lengths;
  for (uint64_t i = 0; i < numFiles; i++) {
    std::string filename(buf + position);
    position += filename.size() + 1;
    LOG(INFO) << "filename: " << filename << ", position: " << position;
    uint64_t size = 0;
    memcpy(&size, buf + position, sizeof(uint64_t));
    position += sizeof(uint64_t);
    offsets[filename] = const_cast<char*>(buf + position);
    lengths[filename] = size;
    position += size;
    CHECK_GE(length, position);
  }

  CHECK_GE(offsets.size(), parameters_.size());

  for (auto& para : parameters_) {
    std::string filename = para->getName();
    if (para->isFullSize()) {
      if (offsets.end() == offsets.find(filename)) {
        para->loadMiss(filename);
      } else {
        std::istringstream stream(
            std::string(offsets[filename], lengths[filename]));
        para->load(stream);
      }
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
