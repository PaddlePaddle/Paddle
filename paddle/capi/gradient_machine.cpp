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

#include "gradient_machine.h"
#include "capi_private.h"
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
paddle_error paddle_gradient_machine_create_for_inference(
    paddle_gradient_machine* machine, void* modelConfigProtobuf, int size) {
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

paddle_error paddle_gradient_machine_create_for_inference_with_parameters(
    paddle_gradient_machine* machine, void* mergedModel, uint64_t size) {
  if (mergedModel == nullptr) return kPD_NULLPTR;
  std::istringstream is(std::string(static_cast<char*>(mergedModel), size));
  int64_t modelConfigSize = 0;
  is.read((char*)(&modelConfigSize), sizeof(modelConfigSize));
  std::string modelConfigProtobuf;
  modelConfigProtobuf.resize(modelConfigSize);
  is.read(&modelConfigProtobuf[0], modelConfigSize);
  paddle::TrainerConfig config;
  paddle::ModelConfig modelConfig;
  if (!config.ParseFromString(modelConfigProtobuf) || !config.IsInitialized()) {
    if (!modelConfig.ParseFromString(modelConfigProtobuf) ||
        !modelConfig.IsInitialized()) {
      return kPD_PROTOBUF_ERROR;
    }
  } else {
    modelConfig = config.model_config();
  }
  auto ptr = new paddle::capi::CGradientMachine();
  ptr->machine.reset(paddle::GradientMachine::create(
      modelConfig, CREATE_MODE_TESTING, {paddle::PARAMETER_VALUE}));
  std::vector<paddle::ParameterPtr>& parameters = ptr->machine->getParameters();
  for (auto& para : parameters) {
    para->load(is);
  }

  *machine = ptr;
  return kPD_NO_ERROR;
}

paddle_error paddle_gradient_machine_destroy(paddle_gradient_machine machine) {
  delete cast(machine);
  return kPD_NO_ERROR;
}

paddle_error paddle_gradient_machine_load_parameter_from_disk(
    paddle_gradient_machine machine, const char* path) {
  auto m = cast(machine);
  if (m == nullptr || path == nullptr || m->machine == nullptr)
    return kPD_NULLPTR;
  m->machine->loadParameters(path);
  return kPD_NO_ERROR;
}

paddle_error paddle_gradient_machine_forward(paddle_gradient_machine machine,
                                             paddle_arguments inArgs,
                                             paddle_arguments outArgs,
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

paddle_error paddle_gradient_machine_create_shared_param(
    paddle_gradient_machine origin,
    void* modelConfigProtobuf,
    int size,
    paddle_gradient_machine* slave) {
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

paddle_error paddle_gradient_machine_randomize_param(
    paddle_gradient_machine machine) {
  auto m = cast(machine);
  if (m == nullptr || m->machine == nullptr) return kPD_NULLPTR;
  m->machine->randParameters();
  return kPD_NO_ERROR;
}

paddle_error paddle_gradient_machine_get_layer_output(
    paddle_gradient_machine machine,
    const char* layerName,
    paddle_arguments args) {
  auto m = cast(machine);
  auto out = paddle::capi::cast<paddle::capi::CArguments>(args);
  if (m == nullptr || layerName == nullptr || out == nullptr ||
      m->machine == nullptr) {
    return kPD_NULLPTR;
  }

  auto layerOutput = m->machine->getLayerOutput(layerName);
  out->args.push_back(layerOutput);
  return kPD_NO_ERROR;
}

paddle_error paddle_gradient_machine_release_layer_output(
    paddle_gradient_machine machine) {
  auto m = cast(machine);
  if (m == nullptr || m->machine == nullptr) {
    return kPD_NULLPTR;
  }
  m->machine->releaseOutput();
  return kPD_NO_ERROR;
}
