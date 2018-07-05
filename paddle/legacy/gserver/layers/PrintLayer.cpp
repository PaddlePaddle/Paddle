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

class PrintLayer : public Layer {
 public:
  explicit PrintLayer(const LayerConfig& config) : Layer(config) {}

  void forward(PassType passType) override {
    Layer::forward(passType);
    std::vector<std::string> vals;
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      std::ostringstream s;
      getInput(i).printValueString(s, "");
      vals.push_back(s.str());
    }
    size_t pos = 0;
    size_t i = 0;
    std::ostringstream s;
    const std::string& format = config_.user_arg();
    while (true) {
      size_t pos1 = format.find("%s", pos);
      if (pos1 == std::string::npos) break;
      if (i >= vals.size()) {
        break;
      }
      s << format.substr(pos, pos1 - pos) << vals[i];
      pos = pos1 + 2;
      ++i;
    }
    if (i != inputLayers_.size()) {
      LOG(ERROR) << "Number of value in the format (" << format
                 << ") is not same as the number of inputs ("
                 << inputLayers_.size() << ") at " << getName();
    }
    s << format.substr(pos);

    const std::string delimiter("\n");
    std::string content = s.str();
    std::string::size_type foundPos = 0;
    std::string::size_type prevPos = 0;
    while ((foundPos = content.find(delimiter, prevPos)) != std::string::npos) {
      LOG(INFO) << content.substr(prevPos, foundPos - prevPos);
      prevPos = foundPos + delimiter.size();
    }
    LOG(INFO) << content.substr(prevPos);
  }

  void backward(const UpdateCallback& callback) override {}
};

REGISTER_LAYER(print, PrintLayer);

}  // namespace paddle
