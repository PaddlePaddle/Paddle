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
 * \brief  This layer pads zeros to inputs according to the specify dimension.
 *         The input and output is a 4D tensor. Padding zeros from the 2nd to
 *         the 4th dimenstion according padc_, padh_ and padw_.
 */
class PadLayer : public Layer {
 public:
  explicit PadLayer(const LayerConfig& config) : Layer(config) {}

  ~PadLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  void setOutDims(const size_t batchSize);
  void setTensorDim(const size_t batchSize);

  std::vector<uint32_t> padc_;
  std::vector<uint32_t> padh_;
  std::vector<uint32_t> padw_;
  TensorShape inDims_;
  TensorShape outDims_;
};
}  // namespace paddle
