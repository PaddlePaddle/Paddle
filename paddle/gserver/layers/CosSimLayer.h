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
 * @brief A layer for calculating cosine similarity between two vector
 * \f[
 * f(x,y)=scale\frac{x_1y_1+x_2y_2+...+x_ny_n}{\sqrt{x_1^2+x_2^2+...
 * +x_n^2}\sqrt{y_1^2+y_2^2+...+y_n^2}}
 * \f]
 *
 * - Input1: A vector (batchSize * dataDim) *
 * - Input2: A vector (batchSize * dataDim) or (1 * dataDim) *
 * - Output: A vector (batchSize * 1)
 *
 * The config file api is cos_sim.
 */
class CosSimLayer : public Layer {
 public:
  explicit CosSimLayer(const LayerConfig& config) : Layer(config) {}

  ~CosSimLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
