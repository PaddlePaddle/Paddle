/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "NormLayer.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief Basic parent layer of normalization
 *
 * @note Normalize the input in local region
 */
class NormLayer : public Layer {
public:
  explicit NormLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    Layer::init(layerMap, parameterMap);
    return true;
  }

  /**
   * @brief create norm layer by norm_type
   */
  static Layer* create(const LayerConfig& config);
};

/**
 * @brief response normalization within feature maps
 * namely normalize in independent channel
 * When code refactoring, we delete the original implementation.
 * Need to implement in the futrue.
 */
class ResponseNormLayer : public NormLayer {
protected:
  size_t channels_, size_, outputX_, imgSize_, outputY_, imgSizeY_;
  real scale_, pow_;
  MatrixPtr denoms_;

public:
  explicit ResponseNormLayer(const LayerConfig& config) : NormLayer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override { LOG(FATAL) << "Not implemented"; }
  void backward(const UpdateCallback& callback = nullptr) override {
    LOG(FATAL) << "Not implemented";
  }
};

}  // namespace paddle
