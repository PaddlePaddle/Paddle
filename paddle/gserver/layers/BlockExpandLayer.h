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
 * @brief Expand feature map to minibatch matrix.
 * - matrix width is: blockH_ * blockW_ * channels_
 * - matirx height is: outputH_ * outputW_
 *
 * \f[
 * outputH\_ = 1 + (2 * paddingH\_ + imgSizeH\_ - blockH\_ + strideH\_ - 1) /
 *             strideH\_ \\
 * outputW\_ = 1 + (2 * paddingW\_ + imgSizeW\_ - blockW\_ + strideW\_ - 1) /
 *             strideW\_
 * \f]
 *
 * The expand method is the same with ExpandConvLayer, but saved the transposed
 * value. After expanding, output_.sequenceStartPositions will store timeline.
 * The number of time steps are outputH_ * outputW_ and the dimension of each
 * time step is blockH_ * blockW_ * channels_. This layer can be used after
 * convolution neural network, and before recurrent neural network.
 *
 * The config file api is block_expand_layer.
 */
class BlockExpandLayer : public Layer {
 protected:
  /**
   * @brief Calculate outputH_ and outputW_ and return block number which
   * actually is time steps.
   * @return time steps, outoutH_ * outputW_.
   */
  size_t getBlockNum();
  size_t blockH_, blockW_, strideH_, strideW_, paddingH_, paddingW_;
  size_t imgSizeH_, imgSizeW_, outputH_, outputW_, channels_;

  TensorShape inputShape_;
  TensorShape outputShape_;

 public:
  explicit BlockExpandLayer(const LayerConfig& config) : Layer(config) {}

  ~BlockExpandLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
