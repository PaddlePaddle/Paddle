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

#include <map>
#include <vector>
#include "DetectionUtil.h"
#include "Layer.h"

namespace paddle {

/**
 * The detection output layer for a SSD detection task. This layer applies the
 * Non-maximum suppression to the all predicted bounding box and keeps the
 * Top-K bounding boxes.
 * - Input: This layer needs three input layers: The first input layer
 *          is the priorbox layer. The rest two input layers are convolution
 *          layers for generating bbox location offset and the classification
 *          confidence.
 * - Output: The predict bounding box locations.
 */

class DetectionOutputLayer : public Layer {
 public:
  explicit DetectionOutputLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr) {}

 protected:
  inline LayerPtr getPriorBoxLayer() { return inputLayers_[0]; }

  inline LayerPtr getLocInputLayer(size_t index) {
    return inputLayers_[1 + index];
  }

  inline LayerPtr getConfInputLayer(size_t index) {
    return inputLayers_[1 + inputNum_ + index];
  }

 private:
  size_t numClasses_;  // number of classes
  size_t inputNum_;    // number of input layers
  real nmsThreshold_;
  real confidenceThreshold_;
  size_t nmsTopK_;
  size_t keepTopK_;
  size_t backgroundId_;

  size_t locSizeSum_;
  size_t confSizeSum_;

  MatrixPtr locBuffer_;
  MatrixPtr confBuffer_;
  MatrixPtr locTmpBuffer_;
  MatrixPtr confTmpBuffer_;
  MatrixPtr priorCpuValue_;
  MatrixPtr locCpuBuffer_;
  MatrixPtr confCpuBuffer_;
};

}  // namespace paddle
