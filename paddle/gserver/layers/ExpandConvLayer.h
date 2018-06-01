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
#include "ConvBaseLayer.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 * This layer expands input and use matrix multiplication to
 * calculate convolution operation.
 *
 * The config file api is img_conv_layer.
 */

class ExpandConvLayer : public ConvBaseLayer {
 public:
  explicit ExpandConvLayer(const LayerConfig& config) : ConvBaseLayer(config) {}

  ~ExpandConvLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

  size_t getOutputSize();

 protected:
  std::vector<TensorShape> inputShape_;
  std::vector<TensorShape> filterShape_;
  std::vector<TensorShape> outputShape_;
};

}  // namespace paddle
