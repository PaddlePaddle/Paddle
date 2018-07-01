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
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {
/**
 * A layer has full connections to all neurons in the previous layer.
 * It computes an inner product with a set of learned weights, and
 * (optionally) adds biases.
 *
 * The config file api is fc_layer.
 */

class FullyConnectedLayer : public Layer {
 protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

 public:
  explicit FullyConnectedLayer(const LayerConfig& config) : Layer(config) {}
  ~FullyConnectedLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  Weight& getWeight(int idx) { return *weights_[idx]; }

  void prefetch() override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
