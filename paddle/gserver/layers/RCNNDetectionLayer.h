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

#include <algorithm>
#include <map>
#include <vector>
#include "DetectionUtil.h"
#include "Layer.h"

namespace paddle {

/**
 * The detection output layer to generate bounding boxes in Fast R-CNN.
 * This layer applies Non-maximum suppression to all predicted bounding
 * boxes and keeps the Top-K bounding-boxes.
 * - Input: This layer needs three input layers: The first input layer
 *          contains the prior-box data. The rest two input layers are
 *          layers for generating bounding-box location offset and the
 *          classification confidence.
 * - Output: The predict bounding boxes.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 */

class RCNNDetectionLayer : public Layer {
public:
  explicit RCNNDetectionLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr) {}

private:
  real nmsThreshold_;
  real confidenceThreshold_;
  size_t nmsTopK_;
  size_t keepTopK_;
  size_t numClasses_;
  size_t backgroundId_;
};

}  // namespace paddle
