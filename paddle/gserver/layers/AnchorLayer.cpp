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
#include <algorithm>
#include <iterator>
#include <random>
#include <utility>
#include <vector>
#include "paddle/utils/ThreadLocal.h"

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
  featStride_ = anchorConf.feat_stride();
  baseSize_ = anchorConf.base_size();
  allowedBorder_ = anchorConf.allowed_border();
  posOverlapThreshold_ = anchorConf.pos_overlap_threshold();
  negOverlapThreshold_ = anchorConf.neg_overlap_threshold();
  rpnBatchSize_ = anchorConf.rpn_batch_size();
  rpnFgRatio_ = anchorConf.rpn_fg_ratio();

  generateBaseAnchors();

  setOutput("rpn_bbox_labels", &bboxLabelsOutput_);
  setOutput("rpn_bbox_targets", &bboxTargetsOutput_);

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

void AnchorLayer::generateAllAnchors(size_t layerHeight, size_t layerWidth) {
  size_t step = layerHeight * layerWidth;
  int shiftX[step], shiftY[step];
  for (size_t i = 0; i < layerWidth; ++i) {
    for (size_t j = 0; j < layerHeight; ++j) {
      shiftX[i * layerWidth + j] = j * featStride_;
      shiftY[i * layerWidth + j] = i * featStride_;
    }
  }

  allAnchors_.reserve(anchors_.size() * step);
  for (size_t i = 0; i < anchors_.size(); ++i) {
    for (size_t j = 0; j < step; ++j) {
      real startX = anchors_[i][0] + shiftX[j];
      real startY = anchors_[i][1] + shiftY[j];
      real endX = anchors_[i][2] + shiftX[j];
      real endY = anchors_[i][3] + shiftY[j];
      std::vector<real> anchor{startX, startY, endX, endY};
      allAnchors_.push_back(anchor);
    }
  }
}

void AnchorLayer::bboxOverlaps(const std::vector<std::vector<real>>& gtBBoxes,
                               std::vector<real>& overlaps) {
  for (size_t i = 0; i < allAnchors_.size(); ++i) {
    if (!(allAnchors_[i][0] + allowedBorder_ >= 0 &&
          allAnchors_[i][1] + allowedBorder_ >= 0 &&
          allAnchors_[i][2] < imageWidth_ + allowedBorder_ &&
          allAnchors_[i][3] <
              imageHeight_ + allowedBorder_)) {  // keep only inside anchors
      continue;
    }
    for (size_t j = 0; j < gtBBoxes.size(); ++j) {
      real width = std::min(allAnchors_[i][2], gtBBoxes[j][2]) -
                   std::max(allAnchors_[i][0], gtBBoxes[j][0]) + 1;
      real height = std::min(allAnchors_[i][3], gtBBoxes[j][3]) -
                    std::max(allAnchors_[i][1], gtBBoxes[j][1]) + 1;
      if (width > 0 && height > 0) {
        real gtboxArea = (gtBBoxes[i][2] - gtBBoxes[i][0] + 1) *
                         (gtBBoxes[i][3] - gtBBoxes[i][1] + 1);
        real anchorArea = (allAnchors_[i][2] - allAnchors_[i][0] + 1) *
                          (allAnchors_[i][3] - allAnchors_[i][1] + 1);
        real overlapArea = width * height;
        overlaps[i * gtBBoxes.size() + j] =
            overlapArea / (gtboxArea + anchorArea - overlapArea);
      }
    }
  }
}

std::pair<size_t, size_t> AnchorLayer::labelAnchors(
    const std::vector<std::vector<real>>& gtBBoxes,
    const std::vector<real>& overlaps,
    std::vector<size_t>& anchorMaxIdxs,
    std::vector<size_t>& gtBBoxMaxIdxs,
    std::vector<int>& labels) {
  size_t numPos = 0;
  size_t numNeg = 0;
  for (size_t n = 0; n < overlaps.size(); ++n) {
    size_t anchorIdx = n / gtBBoxes.size();
    size_t gtBBoxIdx = n % gtBBoxes.size();
    if (overlaps[n] >
        overlaps[anchorIdx * gtBBoxes.size() + anchorMaxIdxs[anchorIdx]]) {
      anchorMaxIdxs[anchorIdx] = gtBBoxIdx;
    }
    if (overlaps[n] >
        overlaps[gtBBoxMaxIdxs[gtBBoxIdx] * gtBBoxes.size() + gtBBoxIdx]) {
      gtBBoxMaxIdxs[gtBBoxIdx] = anchorIdx;
    }
  }
  for (size_t n = 0; n < gtBBoxMaxIdxs.size();
       ++n) {  // fg label: for each gt, anchor with highest overlap
    if (overlaps[gtBBoxMaxIdxs[n] * gtBBoxes.size() + n] > 0) {
      labels[gtBBoxMaxIdxs[n]] = 1;
    }
  }
  for (size_t n = 0; n < anchorMaxIdxs.size();
       ++n) {  // fg/bg/disabled label: above/below threshold IOU
    if (overlaps[n * gtBBoxes.size() + anchorMaxIdxs[n]] >=
        posOverlapThreshold_) {
      labels[n] = 1;
    } else if (overlaps[n * gtBBoxes.size() + anchorMaxIdxs[n]] <=
               negOverlapThreshold_) {
      if (overlaps[n * gtBBoxes.size() + anchorMaxIdxs[n]] < 0) {
        labels[n] = -1;
      } else {
        labels[n] = 0;
      }
    }
  }
  for (size_t n = 0; n < labels.size(); ++n) {
    if (labels[n] == 1) {
      ++numPos;
    } else if (labels[n] == 0) {
      ++numNeg;
    }
  }
  return std::make_pair(numPos, numNeg);
}

void AnchorLayer::targetAnchors(const std::vector<std::vector<real>>& gtBBoxs,
                                const std::vector<size_t>& anchorMaxIdxs,
                                const std::vector<int>& labels,
                                std::vector<real>& targets) {
  for (size_t n = 0; n < allAnchors_.size(); ++n) {
    if (labels[n] == 1) {
      std::vector<real> anchor = anchor2whctr(allAnchors_[n]);
      std::vector<real> gtBBox = anchor2whctr(gtBBoxs[anchorMaxIdxs[n]]);
      targets[n * 4 + 0] = (gtBBox[2] - anchor[2]) / anchor[0];
      targets[n * 4 + 1] = (gtBBox[3] - anchor[3]) / anchor[1];
      targets[n * 4 + 2] = std::log(gtBBox[0] / anchor[0]);
      targets[n * 4 + 3] = std::log(gtBBox[1] / anchor[1]);
    }
  }
}

template <typename T>
void AnchorLayer::sampleAnchors(
    std::vector<T>& allLabels, T label, T disabledLable, size_t m, size_t n) {
  auto& randEngine = ThreadLocalRandomEngine::get();
  for (size_t i = 0; i < allLabels.size(); ++i) {
    if (allLabels[i] == label) {
      if (rand_(randEngine) * n < m) {
        --m;
      } else {
        allLabels[i] = disabledLable;
      }
      --n;
    }
  }
}

void AnchorLayer::forward(PassType passType) {
  Layer::forward(passType);

  auto featMap = getInput(0);  // data from the last feature map
  size_t batchSize = featMap.getBatchSize();
  size_t layerWidth = featMap.getFrameWidth();
  size_t layerHeight = featMap.getFrameHeight();

  auto image = getInput(1);  // data from raw images
  size_t imageWidth = image.getFrameWidth();
  size_t imageHeight = image.getFrameHeight();

  if (imageHeight_ != imageHeight ||
      imageWidth_ != imageWidth) {  // share anchors for the same size
    imageHeight_ = imageHeight;
    imageWidth_ = imageWidth;
    allAnchors_.clear();
    generateAllAnchors(layerHeight, layerWidth);
  }

  auto label = getInput(2);  // sequence data of ground-truth boxes
  MatrixPtr gtValue = getInputValue(2);
  const int* gtStartPosPtr = label.sequenceStartPositions->getData(false);
  size_t seqNum = label.getNumSequences();
  std::vector<real> allLabels;
  std::vector<real> allTargets;
  size_t totalPos = 0;
  size_t totalNeg = 0;
  for (size_t n = 0; n < batchSize; ++n) {
    size_t numGTBBoxes = 0;
    std::vector<std::vector<real>> imgGTBBoxes;
    if (n < seqNum) numGTBBoxes = gtStartPosPtr[n + 1] - gtStartPosPtr[n];
    auto startPos = gtValue->getData() + gtStartPosPtr[n] * 4;
    for (size_t i = 0; i < numGTBBoxes; ++i) {
      std::vector<real> gtBBox;
      gtBBox.push_back(*(startPos + i * 4 + 0));
      gtBBox.push_back(*(startPos + i * 4 + 1));
      gtBBox.push_back(*(startPos + i * 4 + 2));
      gtBBox.push_back(*(startPos + i * 4 + 3));
      imgGTBBoxes.push_back(std::move(gtBBox));
    }

    std::vector<real> overlaps(allAnchors_.size() * imgGTBBoxes.size(),
                               -1);  // init with -1 to label disabled anchors
    bboxOverlaps(imgGTBBoxes,
                 overlaps);  // calculate the overlaps of anchors an gtBBoxes

    std::vector<int> labels(allAnchors_.size(),
                            -1);  // init with -1 to label disabled anchors
    std::vector<size_t> anchorMaxIdxs(
        allAnchors_.size(), 0);  // gtBBox index with max overlap of each anchor
    std::vector<size_t> gtBBoxMaxIdxs(
        imgGTBBoxes.size(), 0);  // anchor index with max overlap of each gtBBox
    std::pair<size_t, size_t> numLabels =
        labelAnchors(imgGTBBoxes,
                     overlaps,
                     anchorMaxIdxs,
                     gtBBoxMaxIdxs,
                     labels);  // lable the anchors
    totalPos += numLabels.first;
    totalNeg += numLabels.second;

    std::vector<real> targets(allAnchors_.size() * 4, 0);
    targetAnchors(imgGTBBoxes,
                  anchorMaxIdxs,
                  labels,
                  targets);  // calculate the targets for bbox regression

    std::copy(labels.begin(), labels.end(), std::back_inserter(allLabels));
    std::copy(targets.begin(), targets.end(), std::back_inserter(allTargets));
  }

  size_t numPos = rpnBatchSize_ * rpnFgRatio_;
  if (totalPos > numPos) {  // subsample positive labels if we have too many
    sampleAnchors<real>(allLabels, 1, -1, numPos, totalPos);
  }
  size_t numNeg = rpnBatchSize_ - numPos;
  if (totalNeg > numNeg) {  // subsample negative labels if we have too many
    sampleAnchors<real>(allLabels, 0, -1, numNeg, totalNeg);
  }

  Layer::resetSpecifyOutput(
      bboxLabelsOutput_, allLabels.size(), 1, false, false);
  bboxLabelsOutput_.value->copyFrom(&allLabels[0], allLabels.size());
  Layer::resetSpecifyOutput(
      bboxTargetsOutput_, allLabels.size(), 4, false, false);
  bboxTargetsOutput_.value->copyFrom(&allTargets[0], allTargets.size());
}

}  // namespace paddle
