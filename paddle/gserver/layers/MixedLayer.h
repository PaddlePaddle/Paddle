/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include "Projection.h"
#include "Operator.h"

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

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  virtual void prefetch();
  virtual void forward(PassType passType);
  virtual void backward(const UpdateCallback& callback = nullptr);
  virtual void resetState();
  /**
   * setState() should be called after getState(). 
   * Argument state consists of all projections states.
   */
  virtual void setState(LayerStatePtr state);
  /**
   * Return state which consists of all projections states.
   */
  virtual LayerStatePtr getState();

protected:
  std::vector<std::unique_ptr<Projection>> projections_;
  std::vector<std::unique_ptr<Operator>> operators_;
  /// the matrix size of projection state
  std::vector<int> projectionStateMatrixSize_;
  std::unique_ptr<Weight> biases_;
  bool sharedBias_;
};
}  // namespace paddle
