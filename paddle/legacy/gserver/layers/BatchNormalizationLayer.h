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

#include "BatchNormBaseLayer.h"
#include "Layer.h"

namespace paddle {

/**
 * @brief A Inheritance class of Batch normalization layer.
 * It supports both CPU and GPU.
 *
 * The config file api is batch_norm_layer.
 */

class BatchNormalizationLayer : public BatchNormBaseLayer {
 public:
  explicit BatchNormalizationLayer(const LayerConfig& config)
      : BatchNormBaseLayer(config), firstTest_(true) {}

  ~BatchNormalizationLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  /// Load pre-calculated mean and std.
  void setMeanAndStd();

  /// Calculate mean and std.
  void calMeanAndStd(const MatrixPtr& mat);

  /// Calculate moving mean and variance.
  void calMovingMeanAndVar();

  /// expand a Matrix from batch, channels* imagePixels to
  /// batch * ImagePixels * channels.
  void expandMat(const MatrixPtr& in, MatrixPtr& out);

  /// Shrink a Matrix from  from batch * ImagePixels * channels
  /// to batch, channels* imagePixels.
  void shrinkMat(const MatrixPtr& in, MatrixPtr& out);

  void onPassEnd() override { firstTest_ = true; }

  MatrixPtr tmpMat_, tmpGrad_;
  MatrixPtr expandedIn_, expandedOut_;
  MatrixPtr expandedInGrad_, expandedOutGrad_, inGrad_;
  MatrixPtr normIn_, normInGrad_, meanGrad_, stdGrad_;

  /// Load mean and variance only once flag.
  bool firstTest_;
};

}  // namespace paddle
