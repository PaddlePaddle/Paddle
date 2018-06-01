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
 * @brief A layer for bilinear interpolation which is
 *        used on conv layer output.
 *
 * @note  The config file api is bilinear_interp_layer.
 */
class BilinearInterpLayer : public Layer {
 protected:
  size_t outImgH_, outImgW_;
  size_t inImgH_, inImgW_;
  real ratioH_, ratioW_;
  size_t numChannels_;

 public:
  explicit BilinearInterpLayer(const LayerConfig& config) : Layer(config) {}

  virtual ~BilinearInterpLayer() {}

  size_t getSize();
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
