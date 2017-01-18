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

#include "Layer.h"

namespace paddle {

/**
 * @brief response normalization across feature maps
 * namely normalize in number of size_ channels
 */
class PadLayer : public Layer {
public:
  explicit PadLayer(const LayerConfig& config) : Layer(config) {}

  ~PadLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

protected:
  void setOutDims(const size_t batchSize);
  void setTensorDim(const size_t batchSize);

  std::vector<int> padc_;
  std::vector<int> padh_;
  std::vector<int> padw_;
  TensorShape inDims_;
  TensorShape outDims_;
};
}  // namespace paddle
