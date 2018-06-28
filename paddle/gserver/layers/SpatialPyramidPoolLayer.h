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
#include "PoolProjection.h"
#include "paddle/math/MathUtils.h"
#include "paddle/utils/Logging.h"

namespace paddle {
/**
 * @brief A layer for spatial pyramid pooling on the input image by taking
 * the max, average, etc. within regions, so that the result vector of
 * different sized images are of the same size.
 *
 * The config file api is spp_layer.
 */

class SpatialPyramidPoolLayer : public Layer {
 protected:
  size_t channels_;
  size_t imgSizeW_;
  size_t imgSizeH_;
  size_t pyramidHeight_;
  std::string poolType_;

  std::vector<std::unique_ptr<PoolProjection>> poolProjections_;
  std::vector<Argument> projOutput_;
  std::vector<std::pair<size_t, size_t>> projCol_;

 public:
  explicit SpatialPyramidPoolLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  ProjectionConfig getConfig(size_t sizeX_,
                             size_t sizeY_,
                             size_t channels,
                             size_t pyamidLevel_,
                             std::string& poolType_);
  size_t getSize();

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};
}  // namespace paddle
