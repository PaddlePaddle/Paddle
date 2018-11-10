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

#pragma once

#include "Layer.h"
#include "LinearChainCTC.h"

namespace paddle {

class CTCLayer : public Layer {
 public:
  explicit CTCLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void forwardImp(const Argument& softmaxSeqs, const Argument& labelSeqs);
  void backward(const UpdateCallback& callback) override;
  void backwardImp(const UpdateCallback& callback,
                   const Argument& softmaxSeqs,
                   const Argument& labelSeqs);

 protected:
  size_t numClasses_;
  bool normByTimes_;
  std::vector<LinearChainCTC> ctcs_;
  std::vector<Argument> tmpCpuInput_;
};

}  // namespace paddle
