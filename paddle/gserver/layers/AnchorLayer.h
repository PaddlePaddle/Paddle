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

#include "Layer.h"

namespace paddle {
/**
 * @brief A layer used by Faster R-CNN to generate anchor-box locations.
 * - Input: Two and only two input layer are accepted. The input layer must be
 *          be a data output layer and a convolution output layer.
 * - Output: The anchor-box locations of the input data.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 * Networks
 */

class AnchorLayer : public Layer {
public:
  explicit AnchorLayer(const LayerConfig& config) : Layer(config) {}
  ~AnchorLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override {}

protected:
  size_t baseSize_;
  size_t featStrideX_;
  size_t featStrideY_;
  size_t allowedBorder_;
  std::vector<real> anchorScales_;
  std::vector<real> anchorRatios_;
  std::vector<std::vector<real>> anchors_;

  void generateBaseAnchors();
  std::vector<std::vector<real>> enumRatio(const std::vector<real>& anchor);
  std::vector<std::vector<real>> enumScale(const std::vector<real>& anchor);
  std::vector<real> anchor2whctr(const std::vector<real>& anchor);
  std::vector<real> whctr2anchor(real w, real h, real ctrX, real ctrY);
  void generateAllAnchors(size_t layerHeight,
                          size_t layerWidth,
                          size_t imageHeight,
                          size_t imageWidth);
};

}  // namespace paddle
