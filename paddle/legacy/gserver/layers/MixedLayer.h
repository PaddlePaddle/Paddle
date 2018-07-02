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
#include "Operator.h"
#include "Projection.h"

namespace paddle {

/**
 * A mixed layer has multiple input layers.
 * Each input layer was processed by a Projection or Operator.
 * The results of all projections or Operators are summed together with bias
 * (if configured), and then go through an activation function and dropout
 * (if configured).
 *
 * The config file api is mixed_layer.
 */
class MixedLayer : public Layer {
 public:
  explicit MixedLayer(const LayerConfig& config) : Layer(config) {}

  ~MixedLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void prefetch() override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
  void resetState() override;
  /**
   * setState() should be called after getState().
   * Argument state consists of all projections states.
   */
  void setState(LayerStatePtr state) override;
  /**
   * Return state which consists of all projections states.
   */
  LayerStatePtr getState() override;

 protected:
  std::vector<std::unique_ptr<Projection>> projections_;
  std::vector<std::unique_ptr<Operator>> operators_;
  /// the matrix size of projection state
  std::vector<int> projectionStateMatrixSize_;
  std::unique_ptr<Weight> biases_;
  bool sharedBias_;
};
}  // namespace paddle
