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
 * \brief  This layer calculate softmax in image channel dimension.
 */
class SwitchOrderLayer : public Layer {
 public:
  explicit SwitchOrderLayer(const LayerConfig& config) : Layer(config) {}

  ~SwitchOrderLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
  void setInDims();
  void setOutDims();

 protected:
  std::vector<std::shared_ptr<FunctionBase>> nchw2nhwc_;
  std::vector<std::shared_ptr<FunctionBase>> nhwc2nchw_;
  TensorShape inDims_;
  TensorShape outDims_;
  std::vector<int> heightAxis_;
  std::vector<int> widthAxis_;
  size_t reshapeHeight_;
  size_t reshapeWidth_;
};
}  // namespace paddle
