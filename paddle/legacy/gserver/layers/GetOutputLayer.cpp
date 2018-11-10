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

#include "Layer.h"

namespace paddle {

class GetOutputLayer : public Layer {
 public:
  explicit GetOutputLayer(const LayerConfig& config) : Layer(config) {}

  ~GetOutputLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    if (!Layer::init(layerMap, parameterMap)) return false;
    CHECK_EQ(1U, inputLayers_.size());
    CHECK_NE(inputArgument_[0], "");
    return true;
  }

  void forward(PassType passType) override {
    output_ = getPrev(0)->getOutput(inputArgument_[0]);
  }
  void backward(const UpdateCallback& callback = nullptr) override {}
};

REGISTER_LAYER(get_output, GetOutputLayer);

}  // namespace paddle
