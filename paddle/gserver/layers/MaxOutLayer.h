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
 * A layer to do max out on conv layer output.
 * Input: output of a conv layer.
 * Output: feature map size same as input.  Channel is (input channel) / groups.
 * So the num of channels should be able to devided by groups.
 *
 * The config file api is maxout_layer.
 */

class MaxOutLayer : public Layer {
 protected:
  size_t groups_;
  size_t imgSizeH_, imgSizeW_;
  /// outputChannels_ = channels_ / groups_
  size_t channels_, outputChannels_;
  /// feature length = imgSizeH_ * imgSizeW_
  size_t featLen_;
  IVectorPtr maxoutId_;

 public:
  /// return imgSizeH_ * imgSizeW_ * outputChannels_;
  size_t getSize();

  explicit MaxOutLayer(const LayerConfig& config) : Layer(config) {}
  virtual ~MaxOutLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
