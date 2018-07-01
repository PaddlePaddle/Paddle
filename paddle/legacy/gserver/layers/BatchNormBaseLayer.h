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
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief Batch normalization layer use to normalizes the input to across the
 * batch.
 *
 * By default, calculating global mean and variance statistics via a running
 * average in the training peroid. Then the pre-calculated global mean and
 * variance are used for testing.
 *
 * Moving mean and variance are located in Parameter object when constructing
 * and the calculation will change them. Now we only save global mean and
 * variance of one thread in first node for GPU.
 * But the calculation in CPU is different, because parameters are shared by
 * multiple threads. Here using ShareCpuMatrix with lock to calculate. We
 * still save global mean and variance in first node in CPU when multi machine.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 */

class BatchNormBaseLayer : public Layer {
 public:
  explicit BatchNormBaseLayer(const LayerConfig& config) : Layer(config) {}

  ~BatchNormBaseLayer() {}

  /**
   * @brief Create BatchNorm layer by norm_type, including batch_norm and
   * cudnn_batch_norm. If do not set norm_type, it will automatically select
   * cudnn_batch_norm for GPU and batch_norm for CPU.
   */
  static Layer* create(const LayerConfig& config);

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  /**
   * @brief Calculate feature map size. Some input uses frameHeight and
   * frameWidth to store feature size
   */
  void calFeatureMapSize();

 protected:
  /// Batch normalization scale parameter, which is referred to as gamma in
  /// in original paper.
  std::unique_ptr<Weight> weight_;
  /// Moving average of mean.
  std::unique_ptr<Weight> movingMean_;
  /// Moving average of variance.
  std::unique_ptr<Weight> movingVar_;
  /// Batch normalization bias parameter, which is referred to as beta in
  /// in original paper.
  std::unique_ptr<Weight> biases_;

  /// Save intermediate results computed during the forward pass,
  /// these can then be reused to speed up the backward pass.
  MatrixPtr savedMean_;
  MatrixPtr savedInvVar_;

  /// Height or width of input image feature.
  /// Both of them are 1 if the input is fully-connected layer.
  int imageD_;
  int imageH_;
  int imageW_;
  /// Height * Width.
  int imgPixels_;
  /// Feature dimension. If the input layer is conv layer, it is the channels
  /// of feature map of the conv layer. If the input layer is fully-connected
  /// layer, it is the dimension of fc layer.
  int channels_;
  // if useGlobalStats_ is true, will use the loaded mean and variance.
  // otherwise, calculate mean and variance in this mini-batch.
  bool useGlobalStats_;
  // use to compute moving mean and variance.
  real movingAvgFraction_;
  // Epsilon is a small random noise used in batch normalization for stability.
  real epsilon_;
};

}  // namespace paddle
