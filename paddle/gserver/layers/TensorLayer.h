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
 * @brief TensorLayer takes two input vectors.
 * \f[
 *     y_{i} = x_{1} * W_{i} * x_{2}^{\rm T}, i=0, 1, ...,K-1
 * \f]
 *
 * - \f$x_{1}\f$: the first input, size is M.
 * - \f$x_{2}\f$: the second input, size is N.
 * - y: output, size is K.
 * - \f$y_{i}\f$: i-th element of y.
 * - \f$W_{i}\f$: the i-th learned weight, dimensions: [M, N].
 * - \f$x_{2}^{\rm T}\f$: the transpose of \f$x_{2}\f$.
 *
 * The config file api is tensor_layer.
 */

class TensorLayer : public Layer {
 protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

 public:
  explicit TensorLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  Weight& getWeight(int idx) { return *weights_[idx]; }

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};
}  // namespace paddle
