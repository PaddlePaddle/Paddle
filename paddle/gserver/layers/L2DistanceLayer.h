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

namespace paddle {

/**
 * @brief The layer calculates the l2 distance between two input vectors.
 * \f[
 * f(\bf{x}, \bf{y}) = \sqrt{\sum_{i=1}^D(x_i - y_i)}
 * \f]
 *
 * - Input1: A vector (batchSize * dataDim)
 * - Input2: A vector (batchSize * dataDim)
 * - Output: A vector (batchSize * 1)
 *
 * The configuration api is: l2_distance_layer.
 */

class L2DistanceLayer : public Layer {
 public:
  explicit L2DistanceLayer(const LayerConfig& config) : Layer(config) {}
  ~L2DistanceLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 private:
  // Store the result of subtracting Input2 from Input1 in forward computation,
  // which will be reused in backward computation.
  MatrixPtr inputSub_;
};

}  // namespace paddle
