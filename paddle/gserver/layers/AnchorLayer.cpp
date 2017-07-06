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

#include "AnchorLayer.h"
#include <vector>

namespace paddle {

REGISTER_LAYER(anchor, AnchorLayer);

bool AnchorLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  const AnchorConfig& anchorConf = config_.inputs(0).anchor_conf();
  std::copy(anchorConf.scale_ratio().begin(),
            anchorConf.scale_ratio().end(),
            std::back_inserter(anchorScales_));
  std::copy(anchorConf.aspect_ratio().begin(),
            anchorConf.aspect_ratio().end(),
            std::back_inserter(anchorRatios_));
  baseSize_ = anchorConf.base_size();
  featStrideX_ = anchorConf.feat_stride_x();
  featStrideY_ = anchorConf.feat_stride_y();
  allowedBorder_ = 0;
  return true;
}

void AnchorLayer::generateBaseAnchors() {
  std::vector<real> baseAnchor{
      0, 0, static_cast<real>(baseSize_ - 1), static_cast<real>(baseSize_ - 1)};
  std::vector<std::vector<real>> ratioAnchors = enumRatio(baseAnchor);
  for (size_t i = 0; i < ratioAnchors.size(); ++i) {
    std::vector<std::vector<real>> tmpAnchors = enumScale(ratioAnchors[i]);
    anchors_.insert(anchors_.end(), tmpAnchors.begin(), tmpAnchors.end());
  }
}

std::vector<std::vector<real>> AnchorLayer::enumRatio(
    const std::vector<real>& anchor) {
  std::vector<std::vector<real>> ratioAnchors;
  std::vector<real> whctr = anchor2whctr(anchor);
  real ctrX = whctr[2];
  real ctrY = whctr[3];
  real size = whctr[0] * whctr[1];
  for (size_t i = 0; i < anchorRatios_.size(); ++i) {
    real ratioSize = size / anchorRatios_[i];
    real ratioW = std::round(std::sqrt(ratioSize));
    real ratioH = std::round(ratioW * anchorRatios_[i]);
    ratioAnchors.push_back(whctr2anchor(ratioW, ratioH, ctrX, ctrY));
  }
  return ratioAnchors;
}

std::vector<std::vector<real>> AnchorLayer::enumScale(
    const std::vector<real>& anchor) {
  std::vector<std::vector<real>> scaleAnchors;
  std::vector<real> whctr = anchor2whctr(anchor);
  real w = whctr[0];
  real h = whctr[1];
  real ctrX = whctr[2];
  real ctrY = whctr[3];
  for (size_t i = 0; i < anchorScales_.size(); ++i) {
    real scaleW = w * anchorScales_[i];
    real scaleH = h * anchorScales_[i];
    scaleAnchors.push_back(whctr2anchor(scaleW, scaleH, ctrX, ctrY));
  }
  return scaleAnchors;
}

std::vector<real> AnchorLayer::anchor2whctr(const std::vector<real>& anchor) {
  std::vector<real> whctr;
  whctr.push_back(anchor[2] - anchor[0] + 1);    // w
  whctr.push_back(anchor[3] - anchor[1] + 1);    // h
  whctr.push_back((anchor[2] + anchor[0]) / 2);  // ctrX
  whctr.push_back((anchor[3] + anchor[1]) / 2);  // ctrY
  return whctr;
}

std::vector<real> AnchorLayer::whctr2anchor(real w,
                                            real h,
                                            real ctrX,
                                            real ctrY) {
  std::vector<real> anchor;
  anchor.push_back(ctrX - 0.5 * (w - 1));
  anchor.push_back(ctrY - 0.5 * (h - 1));
  anchor.push_back(ctrX + 0.5 * (w - 1));
  anchor.push_back(ctrY + 0.5 * (h - 1));
  return anchor;
}

void AnchorLayer::generateAllAnchors(size_t layerHeight,
                                     size_t layerWidth,
                                     size_t imageHeight,
                                     size_t imageWidth) {
  auto* tmpPtr = getOutputValue()->getData();
  if (featStrideX_ == 0)
    featStrideX_ = static_cast<real>(imageWidth) / layerWidth;
  if (featStrideY_ == 0)
    featStrideY_ = static_cast<real>(imageHeight) / layerHeight;
  size_t idx = 0;
  for (size_t h = 0; h < layerHeight; ++h) {
    for (size_t w = 0; w < layerWidth; ++w) {
      for (size_t i = 0; i < anchors_.size(); ++i) {
        // xmin, ymin, xmax, ymax, overflow_flag, img_width, img_height.
        tmpPtr[idx++] = anchors_[i][0] + h * featStrideX_;
        tmpPtr[idx++] = anchors_[i][1] + w * featStrideY_;
        tmpPtr[idx++] = anchors_[i][2] + h * featStrideX_;
        tmpPtr[idx++] = anchors_[i][3] + w * featStrideY_;
        if (tmpPtr[idx - 4] + allowedBorder_ >= 0 &&
            tmpPtr[idx - 3] + allowedBorder_ >= 0 &&
            tmpPtr[idx - 2] < imageWidth + allowedBorder_ &&
            tmpPtr[idx - 1] <
                imageHeight + allowedBorder_) {  // keep only inside anchors
          tmpPtr[idx++] = 1;
        } else {
          tmpPtr[idx++] = -1;
        }
        tmpPtr[idx++] = imageWidth;
        tmpPtr[idx++] =
            imageHeight;  // to be used in proposal generation for box cliping
      }
    }
  }
}

void AnchorLayer::forward(PassType passType) {
  Layer::forward(passType);

  auto featMap = getInput(0);
  size_t layerWidth = featMap.getFrameWidth();
  size_t layerHeight = featMap.getFrameHeight();

  auto image = getInput(1);
  size_t imageWidth = image.getFrameWidth();
  size_t imageHeight = image.getFrameHeight();

  int dim = layerHeight * layerWidth * anchorScales_.size() *
            anchorRatios_.size() * 5;
  reserveOutput(1, dim);

  generateBaseAnchors();
  generateAllAnchors(layerHeight, layerWidth, imageHeight, imageWidth);
}

}  // namespace paddle
