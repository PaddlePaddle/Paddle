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
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override {}
};

void PrintLayer::forward(PassType passType) {
  Layer::forward(passType);
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const auto& argu = getInput(i);
    const std::string& name = inputLayers_[i]->getName();
    if (argu.value) {
      std::ostringstream os;
      argu.value->print(os);
      LOG(INFO) << "layer=" << name << " value matrix:\n" << os.str();
    }
    if (argu.ids) {
      std::ostringstream os;
      argu.ids->print(os, argu.ids->getSize());
      LOG(INFO) << "layer=" << name << " ids vector:\n" << os.str();
    }
    if (auto startPos = argu.sequenceStartPositions) {
      std::ostringstream os;
      startPos->getVector(false)->print(os, startPos->getSize());
      LOG(INFO) << "layer=" << name << " sequence pos vector:\n" << os.str();
    }
    if (auto subStartPos = argu.subSequenceStartPositions) {
      std::ostringstream os;
      subStartPos->getVector(false)->print(os, subStartPos->getSize());
      LOG(INFO) << "layer=" << name << " sub-sequence pos vector:\n"
                << os.str();
    }
  }
}

REGISTER_LAYER(print, PrintLayer);

}  // namespace paddle
