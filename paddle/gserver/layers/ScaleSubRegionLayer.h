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
 * \brief  For each instance, this layer can be used to multiply a value to a
 *         specified sub continuous region. By providing start index and end
 *         index for C/H/W, you can specify the location and shape of the
 *         region.
 *
 *         input_0: Input value.
 *         input_1: Indices value to specify the location an shape of the
 *                  region.
 */
class ScaleSubRegionLayer : public Layer {
 public:
  explicit ScaleSubRegionLayer(const LayerConfig& config) : Layer(config) {}

  ~ScaleSubRegionLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

 protected:
  TensorShape shape_;
  TensorShape indicesShape_;
  size_t imgH_;
  size_t imgW_;
  size_t channelsNum_;
  real value_;
};

}  // namespace paddle
