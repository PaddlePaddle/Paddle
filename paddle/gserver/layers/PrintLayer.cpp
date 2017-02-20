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

#include "Layer.h"

namespace paddle {

class PrintLayer : public Layer {
public:
  explicit PrintLayer(const LayerConfig& config) : Layer(config) {}

  void forward(PassType passType) override {
    Layer::forward(passType);
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      getInput(i).printValueString(LOG(INFO),
                                   "layer=" + inputLayers_[i]->getName() + " ");
    }
  }

  void backward(const UpdateCallback& callback) override {}
};

REGISTER_LAYER(print, PrintLayer);

}  // namespace paddle
