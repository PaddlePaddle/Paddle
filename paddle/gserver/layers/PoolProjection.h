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

#include "Projection.h"
#include "paddle/math/MathUtils.h"

namespace paddle {

class PoolProjection : public Projection {
 protected:
  size_t imgSizeY_, imgSize_;
  size_t outputY_, outputX_;
  size_t strideY_, stride_;
  size_t sizeY_, sizeX_;
  int confPaddingY_, confPadding_;
  size_t channels_;
  std::string poolType_;
  bool excludeMode_;

 public:
  PoolProjection(const ProjectionConfig& config,
                 ParameterPtr parameter,
                 bool useGpu);

  static PoolProjection* create(const ProjectionConfig& config,
                                ParameterPtr parameter,
                                bool useGpu);

  const std::string& getPoolType() const { return poolType_; }

  size_t getSize();
};

class MaxPoolProjection : public PoolProjection {
 public:
  MaxPoolProjection(const ProjectionConfig& config,
                    ParameterPtr parameter,
                    bool useGpu)
      : PoolProjection(config, parameter, useGpu) {}

  virtual void forward();
  virtual void backward(const UpdateCallback& callback = nullptr);
};

class AvgPoolProjection : public PoolProjection {
 public:
  AvgPoolProjection(const ProjectionConfig& config,
                    ParameterPtr parameter,
                    bool useGpu)
      : PoolProjection(config, parameter, useGpu) {}

  virtual void forward();
  virtual void backward(const UpdateCallback& callback = nullptr);
};
}  // namespace paddle
