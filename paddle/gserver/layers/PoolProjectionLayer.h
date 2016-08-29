/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include "PoolLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>

namespace paddle {

class PoolProjectionLayer : public PoolLayer {
protected:
  size_t imgSizeH_, imgSizeW_;
  size_t outputH_, outputW_;

public:
  size_t getSize();
  explicit PoolProjectionLayer(const LayerConfig& config) : PoolLayer(config) {}
};

class MaxPoolProjectionLayer : public PoolProjectionLayer {
public:
  explicit MaxPoolProjectionLayer(const LayerConfig& config)
      : PoolProjectionLayer(config) {}

  ~MaxPoolProjectionLayer() {}

  virtual void forward(PassType passType);
  virtual void backward(const UpdateCallback& callback = nullptr);
};

class AvgPoolProjectionLayer : public PoolProjectionLayer {
public:
  explicit AvgPoolProjectionLayer(const LayerConfig& config)
      : PoolProjectionLayer(config) {}

  ~AvgPoolProjectionLayer() {}

  virtual void forward(PassType passType);
  virtual void backward(const UpdateCallback& callback = nullptr);
};
}  // namespace paddle
