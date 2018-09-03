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

namespace paddle {

/**
 * \brief Row Convolution Layer.
 */
class RowConvLayer : public Layer {
 public:
  explicit RowConvLayer(const LayerConfig& config) : Layer(config) {}

  ~RowConvLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  // Row convolution weight, context_lenght_ * fan_out.
  // fan_out is the size of output feature.
  std::unique_ptr<Weight> weight_;

  // The step number to look ahead plus one equals contexLength_.
  size_t contexLength_;
  TensorShape wDims_;
};
}  // namespace paddle
