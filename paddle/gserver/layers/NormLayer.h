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

/**
 * This layer applys normalization across the channels of each sample to a
 * conv layer's output, and scales the output by a group of trainable factors
 * whose dimensions equal to the number of channels.
 * - Input: One and only one input layer are accepted.
 * - Output: The normalized data of the input data.
 * Reference:
 *    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
 *    Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector
 */
class CrossChannelNormLayer : public NormLayer {
 public:
  explicit CrossChannelNormLayer(const LayerConfig& config)
      : NormLayer(config) {}
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  void forward(PassType passType);
  void backward(const UpdateCallback& callback);
  MatrixPtr createSampleMatrix(MatrixPtr data, size_t iter, size_t spatialDim);
  MatrixPtr createSpatialMatrix(MatrixPtr data, size_t iter, size_t spatialDim);

 protected:
  size_t channels_;
  std::unique_ptr<Weight> scale_;
  MatrixPtr scaleDiff_;
  MatrixPtr normBuffer_;
  MatrixPtr dataBuffer_;
  MatrixPtr channelBuffer_;
  MatrixPtr spatialBuffer_;
  MatrixPtr sampleBuffer_;
};

}  // namespace paddle
