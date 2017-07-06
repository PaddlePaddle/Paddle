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
 * @brief A layer used by faster r-cnn to generate anchors.
 * - Input: the last feature map size, image size, ground-truth bounding-boxes.
 * - Output: rpn_labels, rpn_bbox_targets.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 * Networks
 */

class AnchorLayer : public Layer {
public:
  explicit AnchorLayer(const LayerConfig& config)
      : Layer(config), rand_(0, 1) {}
  ~AnchorLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override {}

protected:
  std::vector<std::vector<real>> anchors_;
  std::vector<std::vector<real>> allAnchors_;
  size_t featStride_;
  size_t baseSize_;
  size_t allowedBorder_;
  real posOverlapThreshold_;
  real negOverlapThreshold_;
  size_t rpnBatchSize_;
  real rpnFgRatio_;
  size_t imageHeight_;
  size_t imageWidth_;
  std::vector<real> anchorScales_;
  std::vector<real> anchorRatios_;
  std::uniform_real_distribution<double> rand_;
  Argument bboxLabelsOutput_;
  Argument bboxTargetsOutput_;

  void generateBaseAnchors();
  std::vector<std::vector<real>> enumRatio(const std::vector<real>& anchor);
  std::vector<std::vector<real>> enumScale(const std::vector<real>& anchor);
  std::vector<real> anchor2whctr(const std::vector<real>& anchor);
  std::vector<real> whctr2anchor(real w, real h, real ctrX, real ctrY);
  void generateAllAnchors(size_t layerHeight, size_t layerWidth);
  void bboxOverlaps(const std::vector<std::vector<real>>& gtBBoxes,
                    std::vector<real>& overlaps);
  std::pair<size_t, size_t> labelAnchors(
      const std::vector<std::vector<real>>& gtBBoxes,
      const std::vector<real>& overlaps,
      std::vector<size_t>& anchorMaxIdxs,
      std::vector<size_t>& gtBBoxMaxIdxs,
      std::vector<int>& labels);
  void targetAnchors(const std::vector<std::vector<real>>& gtBBoxs,
                     const std::vector<size_t>& anchorMaxIdxs,
                     const std::vector<int>& labels,
                     std::vector<real>& targets);
  template <typename T>
  void sampleAnchors(
      std::vector<T>& allLabels, T label, T disabledLable, size_t m, size_t n);
};

}  // namespace paddle
