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
 * @brief A layer for data normalization
 * - Input: One and only one input layer is accepted. The input layer must
 *        be DataLayer with dense data type.
 * - Output: The normalization of the input data
 *
 * Reference:
 *    LA Shalabi, Z Shaaban, B Kasasbeh. Data mining: A preprocessing engine
 *
 * Three data normalization methoeds are considered
 * - z-score: y = (x-mean)/std
 * - min-max: y = (x-min)/(max-min)
 * - decimal-scaling: y = x/10^j, where j is the smallest integer such that
 *max(|y|)<1
 */

class DataNormLayer : public Layer {
 public:
  enum NormalizationStrategy { kZScore = 0, kMinMax = 1, kDecimalScaling = 2 };

  explicit DataNormLayer(const LayerConfig& config) : Layer(config) {}

  ~DataNormLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  int mode_;
  std::unique_ptr<Weight> weight_;
  MatrixPtr min_;
  MatrixPtr rangeReciprocal_;  // 1/(max-min)
  MatrixPtr mean_;
  MatrixPtr stdReciprocal_;      // 1/std
  MatrixPtr decimalReciprocal_;  // 1/10^j
};
}  // namespace paddle
