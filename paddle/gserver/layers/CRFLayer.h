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

#include <memory>

#include "Layer.h"
#include "LinearChainCRF.h"

namespace paddle {

/**
 * A layer for calculating the cost of sequential conditional random field
 * model.
 * See class LinearChainCRF for the detail of the CRF formulation.
 */
class CRFLayer : public Layer {
 public:
  explicit CRFLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

 protected:
  size_t numClasses_;
  ParameterPtr parameter_;
  std::vector<LinearChainCRF> crfs_;
  LayerPtr weightLayer_;            // weight for each sequence
  std::unique_ptr<Weight> weight_;  // parameters
  real coeff_;                      // weight for the layer
};

}  // namespace paddle
