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

#include <fstream>
#include "paddle/utils/Logging.h"

#include "GradientMachineMode.h"
#include "MultiGradientMachine.h"
#include "MultiNetwork.h"
#include "NeuralNetwork.h"
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

GradientMachine* GradientMachine::create(const std::string& modelFile,
                                         DataConfig* dataConfig) {
  std::ifstream is(modelFile);
  CHECK(is) << "Fail to open " << modelFile;
  return create(is, dataConfig);
}

GradientMachine* GradientMachine::create(std::istream& is,
                                         DataConfig* dataConfig) {
  TrainerConfig trainerConfig;
  GradientMachine* ret = create(is, &trainerConfig);
  if (dataConfig && trainerConfig.has_data_config()) {
    *dataConfig = trainerConfig.data_config();
  }
  return ret;
}

GradientMachine* GradientMachine::create(const std::string& modelFile,
                                         TrainerConfig* trainerConfig) {
  std::ifstream is(modelFile);
  CHECK(is) << "Fail to open " << modelFile;
  return create(is, trainerConfig);
}

GradientMachine* GradientMachine::create(std::istream& is,
                                         TrainerConfig* trainerConfig) {
  TrainerConfig trainerConfigTemp;
  int64_t size;
  CHECK(is.read((char*)&size, sizeof(size))) << "Fail to read ";
  std::string buf;
  buf.resize(size);
  CHECK(is.read(&buf[0], size)) << "Fail to read ";
  CHECK(trainerConfigTemp.ParseFromString(buf)) << "Fail to parse config";
  std::unique_ptr<GradientMachine> machine(
      create(trainerConfigTemp.model_config()));
  std::vector<ParameterPtr>& parameters = machine->getParameters();
  for (auto& para : parameters) {
    para->load(is);
  }

  machine->onLoadParameter();

  if (trainerConfig) {
    *trainerConfig = trainerConfigTemp;
  }

  return machine.release();
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
