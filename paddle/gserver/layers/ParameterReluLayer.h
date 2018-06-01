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
 *  @brief ParameterReluLayer active inputs with learnable parameter weight_.
 *  forward:
 *  \f[
 *      y = x > 0 ? x : w .* x
 *  \f]
 *  backward:
 *  \f[
 *      dx = x > 0 ? dy : w .* dy \\
 *      dw = x > 0 ? 0 : dy.*x
 *  \f]
 *  Here, x is the input, w is the weight, y is the output.
 *  dx, dw, dy is the gradient.
 */

class ParameterReluLayer : public Layer {
 protected:
  std::unique_ptr<Weight> weight_;

  /**
   *  @brief partialSum_ makes a group of inputs share same weights,
   *  - partialSum_ = 1:
   *       element wise activation: each element has a weight_,
   *  - partialSum_ = number of elements in one channel,
   *       channels wise parameter activation, elements in a channel
   *       share same weight_,
   *  - partialSum_ = number of outputs
   *       all elements share same weight_,
   */
  size_t partialSum_;

 public:
  explicit ParameterReluLayer(const LayerConfig& config) : Layer(config) {}

  ~ParameterReluLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};
}  // namespace paddle
