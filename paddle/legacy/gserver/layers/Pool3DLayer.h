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

#include <vector>
#include "Layer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief Basic parent layer of pooling
 * Pools the input within regions
 */
class Pool3DLayer : public Layer {
 public:
  explicit Pool3DLayer(const LayerConfig& config) : Layer(config) {}
  ~Pool3DLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;
  size_t getSize();

 protected:
  int channels_;
  int sizeX_, sizeY_, sizeZ_;
  int strideW_, strideH_, strideD_;
  int paddingW_, paddingH_, paddingD_;
  int imgSizeW_, imgSizeH_, imgSizeD_;
  int outputW_, outputH_, outputD_;
  std::string poolType_;
  MatrixPtr maxPoolIdx_;
};
}  // namespace paddle
