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

#include <vector>
#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * This layer transpose the pooling process.
 * It takes two input, the first input is the input data, and
 * the second is the mask data from the max-pool-with-mask layer.
 *
 */

class UpsampleLayer : public Layer {
 public:
  explicit UpsampleLayer(const LayerConfig& config) : Layer(config) {}
  ~UpsampleLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

  size_t getOutputSize();

 protected:
  size_t scale_, scaleY_;
  size_t upsampleSize_, upsampleSizeY_;
  size_t padOutX_, padOutY_;
  size_t imgSize_, imgSizeY_;
  size_t channels_;
};

}  // namespace paddle
