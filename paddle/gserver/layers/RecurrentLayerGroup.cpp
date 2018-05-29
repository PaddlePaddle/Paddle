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

#include <functional>
#include "paddle/gserver/layers/Layer.h"

#include "paddle/gserver/gradientmachines/RecurrentGradientMachine.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * Recurrent layer group is a group of layers, which forward/backward one frame
 * after previous frame forward/backward through all layers in layer group.
 * It's automatically added by config_parser if some layers are defined
 * between RecurrentLayerGroupBegin and RecurrentLayerGroupEnd.
 */
class RecurrentLayerGroup : public Layer {
 public:
  explicit RecurrentLayerGroup(const LayerConfig& config) : Layer(config) {}

  void initSubNetwork(NeuralNetwork* rootNetwork,
                      const ModelConfig& config,
                      const std::vector<ParameterType>& parameterTypes,
                      bool useGpu) override;

  void forward(PassType passType) override {
    REGISTER_TIMER_INFO("RecurrentGroupFwTime", getName().c_str());
    const std::vector<Argument> inArgs;
    std::vector<Argument> outArgs;
    network_->forward(inArgs, &outArgs, passType);
  }
  void backward(const UpdateCallback& callback) override {
    REGISTER_TIMER_INFO("RecurrentGroupBwTime", getName().c_str());
    network_->backward(nullptr);

    for (auto& para : parameters_) {
      para->incUpdate(callback);
    }
  }

  /**
   * @see Layer.accessSubNetwork
   */
  void accessSubNetwork(
      const std::function<void(NeuralNetwork&)>& callback) override {
    callback(*network_);
  }

 private:
  std::unique_ptr<RecurrentGradientMachine> network_;
};

REGISTER_LAYER(recurrent_layer_group, RecurrentLayerGroup);

void RecurrentLayerGroup::initSubNetwork(
    NeuralNetwork* rootNetwork,
    const ModelConfig& config,
    const std::vector<ParameterType>& parameterTypes,
    bool useGpu) {
  setNeedGradient(true);

  network_.reset(new RecurrentGradientMachine(config_.name(), rootNetwork));
  ParamInitCallback cb = [rootNetwork](int paramId, Parameter* para) {
    para->enableSharedType(
        PARAMETER_VALUE,
        rootNetwork->getParameters()[paramId]->getBuf(PARAMETER_VALUE),
        rootNetwork->getParameters()[paramId]->getMat(PARAMETER_VALUE));
    para->enableSharedType(
        PARAMETER_GRADIENT,
        rootNetwork->getParameters()[paramId]->getBuf(PARAMETER_GRADIENT),
        rootNetwork->getParameters()[paramId]->getMat(PARAMETER_GRADIENT));
  };
  network_->init(config, cb, parameterTypes, useGpu);

  for (auto paramId : network_->getParameterIds()) {
    ParameterPtr parameter = rootNetwork->getParameters()[paramId];
    parameter->incShared();
    CHECK_EQ(parameter->getDeviceId(), getDeviceId());
    parameters_.push_back(parameter);
  }
}

}  // namespace paddle
