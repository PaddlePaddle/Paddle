/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

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
#include "ConvBaseLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 * This layer expands input and use matrix multiplication to
 * calculate convolution operation.
 */
class Conv3DLayer : public ConvBaseLayer {
 public:
  explicit Conv3DLayer(const LayerConfig& config) : ConvBaseLayer(config) {}
  ~Conv3DLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void addBias();
  void backward(const UpdateCallback& callback);
  void bpropBiases();
  void bpropData(int i);
  void bpropWeights(int i);
  size_t getSize();

 protected:
  // Figure out the dimensions for individual gemms.
  IntV M_;  /// numFilters_ / filter_group_;
  IntV N_;  /// channels_ * filterSizeZ_ * filterSize_ * filterSizeY_
  IntV K_;  /// outputD_ * outputH_ * outputW_
  MatrixPtr colBuf_;
};

}  // namespace paddle
