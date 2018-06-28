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
 * A layer used by Fast R-CNN to extract feature maps of ROIs from the last
 * feature map.
 * - Input: This layer needs two input layers: The first input layer is a
 *          convolution layer; The second input layer contains the ROI data
 *          which is the output of ProposalLayer in Faster R-CNN. layers for
 *          generating bbox location offset and the classification confidence.
 * - Output: The ROIs' feature map.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 * Networks
 */

class ROIPoolLayer : public Layer {
 protected:
  size_t channels_;
  size_t width_;
  size_t height_;
  size_t pooledWidth_;
  size_t pooledHeight_;
  real spatialScale_;

  // Since there is no int matrix, use real maxtrix instead.
  MatrixPtr maxIdxs_;

 public:
  explicit ROIPoolLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};
}  // namespace paddle
