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

#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"

#define cast(v) paddle::capi::cast<paddle::capi::CGradientMachine>(v)

enum GradientMatchineCreateMode {
  CREATE_MODE_NORMAL = 0,
  CREATE_MODE_TESTING = 4
};

namespace paddle {

class MyNeuralNetwork : public NeuralNetwork {
public:
  MyNeuralNetwork(const std::string& name, NeuralNetwork* network)
      : NeuralNetwork(name, network) {}
};

NeuralNetwork* newCustomNerualNetwork(const std::string& name,
                                      NeuralNetwork* network) {
  return new MyNeuralNetwork(name, network);
}
}  // namespace paddle

extern "C" {
int PDGradientMachineCreateForPredict(PD_GradientMachine* machine,
                                      void* modelConfigProtobuf,
                                      int size) {
  if (modelConfigProtobuf == nullptr) return kPD_NULLPTR;
  paddle::ModelConfig config;
  if (!config.ParseFromArray(modelConfigProtobuf, size) ||
      !config.IsInitialized()) {
    return kPD_PROTOBUF_ERROR;
  }

  auto ptr = new paddle::capi::CGradientMachine();
  ptr->machine.reset(paddle::GradientMachine::create(
      config, CREATE_MODE_TESTING, {paddle::PARAMETER_VALUE}));
  *machine = ptr;
  return kPD_NO_ERROR;
}

int PDGradientMachineDestroy(PD_GradientMachine machine) {
  delete cast(machine);
  return kPD_NO_ERROR;
}

int PDGradientMachineLoadParameterFromDisk(PD_GradientMachine machine,
                                           const char* path) {
  auto m = cast(machine);
  if (m == nullptr || path == nullptr || m->machine == nullptr)
    return kPD_NULLPTR;
  m->machine->loadParameters(path);
  return kPD_NO_ERROR;
}

int PDGradientMachineForward(PD_GradientMachine machine,
                             PD_Arguments inArgs,
                             PD_Arguments outArgs,
                             bool isTrain) {
  auto m = cast(machine);
  auto in = paddle::capi::cast<paddle::capi::CArguments>(inArgs);
  auto out = paddle::capi::cast<paddle::capi::CArguments>(outArgs);
  if (m == nullptr || in == nullptr || out == nullptr || m->machine == nullptr)
    return kPD_NULLPTR;
  m->machine->forward(
      in->args, &out->args, isTrain ? paddle::PASS_TRAIN : paddle::PASS_TEST);
  return kPD_NO_ERROR;
}

int PDGradientMachineCreateSharedParam(PD_GradientMachine origin,
                                       void* modelConfigProtobuf,
                                       int size,
                                       PD_GradientMachine* slave) {
  auto o = cast(origin);
  if (origin == nullptr || slave == nullptr || o->machine == nullptr) {
    return kPD_NULLPTR;
  }
  paddle::ModelConfig config;
  if (!config.ParseFromArray(modelConfigProtobuf, size) ||
      !config.IsInitialized()) {
    return kPD_PROTOBUF_ERROR;
  }

  std::unique_ptr<paddle::capi::CGradientMachine> ptr(
      new paddle::capi::CGradientMachine());
  auto nn = paddle::NeuralNetwork::create(config);
  nn->init(config,
           [&o](int paramId, paddle::Parameter* param) {
             auto p = o->machine->getParameters()[paramId];
             param->enableSharedType(paddle::PARAMETER_VALUE,
                                     p->getBuf(paddle::PARAMETER_VALUE));
           },
           {paddle::PARAMETER_VALUE},
           false);
  ptr->machine.reset(nn);
  *slave = ptr.release();
  return kPD_NO_ERROR;
}
}
