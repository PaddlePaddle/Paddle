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
#include "PoolLayer.h"
#include "PoolProjection.h"
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * @brief Basic parent layer of different kinds of pooling
 */
class PoolProjectionLayer : public PoolLayer {
 protected:
  size_t imgSizeH_, imgSizeW_;
  size_t outputH_, outputW_;
  std::unique_ptr<PoolProjection> poolProjection_;
  ProjectionConfig projectionConfig_;

 public:
  explicit PoolProjectionLayer(const LayerConfig& config) : PoolLayer(config) {
    PoolConfig* conf = projectionConfig_.mutable_pool_conf();
    *conf = config_.inputs(0).pool_conf();
    poolProjection_.reset(
        PoolProjection::create(projectionConfig_, nullptr, useGpu_));
  }

  size_t getSize();

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};
}  // namespace paddle
