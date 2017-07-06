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

#include "RPNLossLayer.h"
#include <float.h>
#include <vector>
#include "DataLayer.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

REGISTER_LAYER(rpn_loss, RPNLossLayer);

bool RPNLossLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  auto layerConf = config_.inputs(0).rpn_loss_conf();
  posOverlapThreshold_ = layerConf.pos_overlap_threshold();
  negOverlapThreshold_ = layerConf.neg_overlap_threshold();
  rpnBatchSize_ = layerConf.rpn_batch_size();
  rpnFgRatio_ = layerConf.rpn_fg_ratio();
  lossRatio_ = layerConf.loss_ratio();
  numClasses_ = 2;
  inputNum_ = 1;
  backgroundId_ = 0;

  return true;
}

void RPNLossLayer::bboxOverlaps(
    const std::vector<std::vector<real>>& anchorBoxes,
    const std::vector<std::vector<real>>& gtBBoxes,
    std::vector<real>& overlaps) {
  for (size_t i = 0; i < anchorBoxes.size(); ++i) {
    if (anchorBoxes[i][4] == -1) {
      continue;  // out of the image, keep only inside anchors
    }
    for (size_t j = 0; j < gtBBoxes.size(); ++j) {
      real width = std::min(anchorBoxes[i][2], gtBBoxes[j][2]) -
                   std::max(anchorBoxes[i][0], gtBBoxes[j][0]) + 1;
      real height = std::min(anchorBoxes[i][3], gtBBoxes[j][3]) -
                    std::max(anchorBoxes[i][1], gtBBoxes[j][1]) + 1;
      if (width > 0 && height > 0) {
        real gtboxArea = (gtBBoxes[j][2] - gtBBoxes[j][0] + 1) *
                         (gtBBoxes[j][3] - gtBBoxes[j][1] + 1);
        real anchorArea = (anchorBoxes[i][2] - anchorBoxes[i][0] + 1) *
                          (anchorBoxes[i][3] - anchorBoxes[i][1] + 1);
        real overlapArea = width * height;
        overlaps[i * gtBBoxes.size() + j] =
            overlapArea / (gtboxArea + anchorArea - overlapArea);
      }
    }
  }
}

std::pair<size_t, size_t> RPNLossLayer::labelAnchors(
    const std::vector<std::vector<real>>& anchorBoxes,
    const std::vector<std::vector<real>>& gtBBoxes,
    const std::vector<real>& overlaps,
    const real posOverlapThreshold,
    const real negOverlapThreshold,
    std::vector<int>& matchIndices,
    std::vector<int>& labels) {
  size_t numPos = 0;
  size_t numNeg = 0;
  std::vector<int> gtBBoxMaxIdxs(
      gtBBoxes.size(), -1);  // anchor index with max overlap of each gtBBox
  for (size_t n = 0; n < overlaps.size(); ++n) {
    size_t anchorIdx = n / gtBBoxes.size();
    size_t gtBBoxIdx = n % gtBBoxes.size();
    if (matchIndices[anchorIdx] == -1 ||
        overlaps[n] >
            overlaps[anchorIdx * gtBBoxes.size() + matchIndices[anchorIdx]]) {
      matchIndices[anchorIdx] = gtBBoxIdx;  // overlaps.argmax(axis=1)
    }
    if (gtBBoxMaxIdxs[gtBBoxIdx] == -1 ||
        overlaps[n] >
            overlaps[gtBBoxMaxIdxs[gtBBoxIdx] * gtBBoxes.size() + gtBBoxIdx]) {
      gtBBoxMaxIdxs[gtBBoxIdx] = anchorIdx;  // overlaps.argmax(axis=0)
    }
  }
  for (size_t n = 0; n < gtBBoxMaxIdxs.size();
       ++n) {  // fg label: anchor with highest overlap for each gtBBox
    if (overlaps[gtBBoxMaxIdxs[n] * gtBBoxes.size() + n] > 0) {
      labels[gtBBoxMaxIdxs[n]] = 1;
    }
  }
  for (size_t n = 0; n < anchorBoxes.size();
       ++n) {  // fg/bg/disabled label: above/below threshold IOU
    if (overlaps[n * gtBBoxes.size() + matchIndices[n]] >=
        posOverlapThreshold) {
      labels[n] = 1;
    } else if (overlaps[n * gtBBoxes.size() + matchIndices[n]] <=
               negOverlapThreshold) {
      if (overlaps[n * gtBBoxes.size() + matchIndices[n]] < 0) {
        labels[n] = -1;  // out of the image
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

template <typename T>
void RPNLossLayer::sampleAnchors(
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

pair<size_t, size_t> RPNLossLayer::generateMatchIndices(
    const Matrix& priorValue,
    const size_t numPriorBBoxes,
    const Matrix& gtValue,
    const int* gtStartPosPtr,
    const size_t seqNum,
    const size_t batchSize,
    const real posOverlapThreshold,
    const real negOverlapThreshold,
    const size_t boxBatchSize,
    const real boxFgRatio,
    std::vector<std::vector<int>>* matchIndicesVecPtr,
    std::vector<std::vector<int>>* negIndicesVecPtr) {
  size_t totalPos = 0;
  size_t totalNeg = 0;
  std::vector<real> allLabels;
  std::vector<real> allTargets;

  std::vector<std::vector<real>> anchorBoxes;
  const real* priorData = priorValue.getData();
  for (size_t n = 0; n < numPriorBBoxes; ++n) {
    std::vector<real> anchorBox;
    anchorBox.push_back(*(priorData + n * 7 + 0));
    anchorBox.push_back(*(priorData + n * 7 + 1));
    anchorBox.push_back(*(priorData + n * 7 + 2));
    anchorBox.push_back(*(priorData + n * 7 + 3));
    anchorBox.push_back(*(priorData + n * 7 + 4));
    anchorBoxes.push_back(anchorBox);
  }

  for (size_t n = 0; n < batchSize; ++n) {
    std::vector<int> matchIndices;
    std::vector<int> negIndices;
    matchIndices.resize(numPriorBBoxes, -1);
    size_t numGTBBoxes = 0;
    if (n < seqNum) numGTBBoxes = gtStartPosPtr[n + 1] - gtStartPosPtr[n];
    if (!numGTBBoxes) {
      matchIndicesVecPtr->push_back(matchIndices);
      negIndicesVecPtr->push_back(negIndices);
      continue;
    }
    std::vector<std::vector<real>> gtBBoxes;
    if (n < seqNum) numGTBBoxes = gtStartPosPtr[n + 1] - gtStartPosPtr[n];
    auto startPos = gtValue.getData() + gtStartPosPtr[n] * 4;
    for (size_t i = 0; i < numGTBBoxes; ++i) {
      std::vector<real> gtBBox;
      gtBBox.push_back(*(startPos + i * 4 + 0));
      gtBBox.push_back(*(startPos + i * 4 + 1));
      gtBBox.push_back(*(startPos + i * 4 + 2));
      gtBBox.push_back(*(startPos + i * 4 + 3));
      gtBBoxes.push_back(gtBBox);
    }

    std::vector<real> overlaps(anchorBoxes.size() * gtBBoxes.size(),
                               -1);  // init with -1 to label disabled anchors
    bboxOverlaps(anchorBoxes,
                 gtBBoxes,
                 overlaps);  // calculate the overlaps of anchors and gtBBoxes

    std::vector<int> labels(anchorBoxes.size(),
                            -1);  // init with -1 to label disabled anchors
    std::pair<size_t, size_t> numLabels =
        labelAnchors(anchorBoxes,
                     gtBBoxes,
                     overlaps,
                     posOverlapThreshold,
                     negOverlapThreshold,
                     matchIndices,
                     labels);  // lable the anchors
    totalPos += numLabels.first;
    totalNeg += numLabels.second;
    matchIndicesVecPtr->push_back(matchIndices);
    std::copy(labels.begin(), labels.end(), std::back_inserter(allLabels));
  }

  size_t numPos = boxBatchSize * boxFgRatio;
  if (totalPos > numPos) {  // subsample positive labels if we have too many
    sampleAnchors<real>(allLabels, 1, -1, numPos, totalPos);
  }
  size_t numNeg = boxBatchSize - numPos;
  if (totalNeg > numNeg) {  // subsample negative labels if we have too many
    sampleAnchors<real>(allLabels, 0, -1, numNeg, totalNeg);
  }

  for (size_t n = 0; n < batchSize; ++n) {
    std::vector<int> negIndices;
    for (size_t i = 0; i < numPriorBBoxes; ++i) {
      size_t idx = n * numPriorBBoxes + i;
      if (allLabels[idx] != 1) {
        (*matchIndicesVecPtr)[n][i] = -1;
        if (allLabels[idx] == 0) {
          negIndices.push_back(i);
        }
      }
    }
    negIndicesVecPtr->push_back(negIndices);
  }

  return std::make_pair(numPos, numNeg);
}

void RPNLossLayer::encodeTarget(const std::vector<real>& anchorBox,
                                const std::vector<real>& gtBBox,
                                std::vector<real>& target) {
  real anchorBoxWidth = anchorBox[2] - anchorBox[0] + 1;
  real anchorBoxHeight = anchorBox[3] - anchorBox[1] + 1;
  real anchorBoxCenterX = (anchorBox[2] + anchorBox[0]) / 2;
  real anchorBoxCenterY = (anchorBox[3] + anchorBox[1]) / 2;

  real gtBBoxWidth = gtBBox[2] - gtBBox[0] + 1;
  real gtBBoxHeight = gtBBox[3] - gtBBox[1] + 1;
  real gtBBoxCenterX = (gtBBox[2] + gtBBox[0]) / 2;
  real gtBBoxCenterY = (gtBBox[3] + gtBBox[1]) / 2;

  target[0] = (gtBBoxCenterX - anchorBoxCenterX) / anchorBoxWidth;
  target[1] = (gtBBoxCenterY - anchorBoxCenterY) / anchorBoxHeight;
  target[2] = std::log(gtBBoxWidth / anchorBoxWidth);
  target[3] = std::log(gtBBoxHeight / anchorBoxHeight);
}

void RPNLossLayer::forward(PassType passType) {
  Layer::forward(passType);
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();
  resetOutput(batchSize, 1);

  // all location data and confidence score data
  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; ++n) {  // there is only one for RPN
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    locSizeSum_ += inLoc->getElementCnt();
    confSizeSum_ += inConf->getElementCnt();
  }

  // locBuffer layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin2 ......
  Matrix::resizeOrCreate(locTmpBuffer_, 1, locSizeSum_, false, useGpu_);
  locBuffer_ = locTmpBuffer_;

  // confBuffer layout:
  // | class1 score | class2 score | ... |classN score | class1 score | ......
  Matrix::resizeOrCreate(confTmpBuffer_, 1, confSizeSum_, false, useGpu_);
  confBuffer_ = confTmpBuffer_;

  // concate location data and confidence score data
  size_t locOffset = 0;
  size_t confOffset = 0;
  auto& layerConf = config_.inputs(0).rpn_loss_conf();
  for (size_t n = 0; n < inputNum_; ++n) {  // there is only one for RPN
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    if (!height) height = layerConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = layerConf.width();
    locOffset += appendWithPermute(*inLoc,
                                   height,
                                   width,
                                   locSizeSum_,
                                   locOffset,
                                   batchSize,
                                   *locBuffer_,
                                   kNCHWToNHWC);
    confOffset += appendWithPermute(*inConf,
                                    height,
                                    width,
                                    confSizeSum_,
                                    confOffset,
                                    batchSize,
                                    *confBuffer_,
                                    kNCHWToNHWC);
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);

  // priorValue layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | overflow_flag | img_width | img_height |
  // | xmin2 | ......
  MatrixPtr priorValue;

  // labelValue layout:
  // | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | ......
  MatrixPtr labelValue;

  // Copy data from GPU to CPU if use GPU
  if (useGpu_) {
    Matrix::resizeOrCreate(locCpuBuffer_, 1, locSizeSum_, false, false);
    Matrix::resizeOrCreate(confCpuBuffer_, 1, confSizeSum_, false, false);
    MatrixPtr priorTmpValue = getInputValue(*getPriorBoxLayer());
    Matrix::resizeOrCreate(
        priorCpuValue_, 1, priorTmpValue->getElementCnt(), false, false);
    MatrixPtr labelTmpValue = getInputValue(*getLabelLayer());
    Matrix::resizeOrCreate(labelCpuValue_,
                           labelTmpValue->getHeight(),
                           labelTmpValue->getWidth(),
                           false,
                           false);

    locCpuBuffer_->copyFrom(*locTmpBuffer_);
    confCpuBuffer_->copyFrom(*confTmpBuffer_);
    priorCpuValue_->copyFrom(*priorTmpValue);
    labelCpuValue_->copyFrom(*labelTmpValue);

    locBuffer_ = locCpuBuffer_;
    confBuffer_ = confCpuBuffer_;
    priorValue = priorCpuValue_;
    labelValue = labelCpuValue_;
  } else {
    priorValue = getInputValue(*getPriorBoxLayer());
    labelValue = getInputValue(*getLabelLayer());
  }

  // Match anchor-box to groundtruth bbox
  Argument label = getInput(*getLabelLayer());
  const int* labelIndex = label.sequenceStartPositions->getData(false);
  size_t seqNum = label.getNumSequences();
  numMatches_ = 0;
  numNegs_ = 0;
  allMatchIndices_.clear();
  allNegIndices_.clear();
  numPriors_ = priorValue->getElementCnt() / 7;

  std::pair<size_t, size_t> retPair = generateMatchIndices(*priorValue,
                                                           numPriors_,
                                                           *labelValue,
                                                           labelIndex,
                                                           seqNum,
                                                           batchSize,
                                                           posOverlapThreshold_,
                                                           negOverlapThreshold_,
                                                           rpnBatchSize_,
                                                           rpnFgRatio_,
                                                           &allMatchIndices_,
                                                           &allNegIndices_);
  numMatches_ = retPair.first;
  numNegs_ = retPair.second;

  // BBox location L1 smooth loss
  locLoss_ = 0.0;
  if (numMatches_ >= 1) {
    size_t count = 0;
    MatrixPtr locLossOutput;
    Matrix::resizeOrCreate(locLossOutput, numMatches_ * 4, 1, false, false);
    Matrix::resizeOrCreate(locGTData_, numMatches_ * 4, 1, false, false);
    Matrix::resizeOrCreate(locDiff_, numMatches_ * 4, 1, false, false);
    locDiff_->zeroMem();
    std::vector<real> locGTData;

    real* locDiffData = locDiff_->getData();
    const real* locBufferData = locBuffer_->getData();
    for (size_t n = 0; n < batchSize; ++n) {
      for (size_t i = 0; i < numPriors_; ++i) {
        if (allMatchIndices_[n][i] == -1) continue;  // match none
        size_t locOffset =
            n * (locBuffer_->getElementCnt() / batchSize) + i * 4;
        std::copy(locBufferData + locOffset,
                  locBufferData + locOffset + 4,
                  locDiffData + count);
        count += 4;
        const int gtIdx = allMatchIndices_[n][i];
        auto* priorOffset = priorValue->getData() + i * 7;
        std::vector<real> anchorBox{
            *(priorOffset + 0),
            *(priorOffset + 1),
            *(priorOffset + 2),
            *(priorOffset + 3),
        };
        auto* labelOffset = labelValue->getData() + (labelIndex[n] + gtIdx) * 4;
        std::vector<real> gtBBox{
            *(labelOffset + 0),
            *(labelOffset + 1),
            *(labelOffset + 2),
            *(labelOffset + 3),
        };
        std::vector<real> gtEncode(4);
        encodeTarget(anchorBox, gtBBox, gtEncode);
        locGTData.insert(locGTData.end(), gtEncode.begin(), gtEncode.end());
      }
    }
    locGTData_->copyFrom(&locGTData[0], numMatches_ * 4);
    locLossOutput->smoothL1(*locDiff_, *locGTData_, 0.0);
    locLoss_ = locLossOutput->getSum() / numMatches_ * lossRatio_;
  }

  // BBox confidence softmax loss
  confLoss_ = 0;
  numConf_ = numMatches_ + numNegs_;
  if (numConf_ >= 1) {
    Matrix::resizeOrCreate(confProb_, numConf_, numClasses_, false, false);
    IVector::resizeOrCreate(confGTData_, numConf_, false);
    confProb_->zeroMem();
    size_t count = 0;

    std::vector<real> confPredData;
    real* confProbData = confProb_->getData();
    const real* confBufferData = confBuffer_->getData();
    for (size_t n = 0; n < batchSize; ++n) {
      for (size_t i = 0; i < numPriors_; ++i) {
        if (allMatchIndices_[n][i] == -1) continue;
        confGTData_->getData()[count] = 1;
        size_t confOffset = n * numPriors_ * numClasses_ + i * numClasses_;
        std::copy(confBufferData + confOffset,
                  confBufferData + confOffset + numClasses_,
                  confProbData + count * numClasses_);
        confPredData.reserve(confPredData.size() + numClasses_);
        confPredData.insert(confPredData.end(),
                            confBufferData + confOffset,
                            confBufferData + confOffset + numClasses_);
        ++count;
      }
      for (size_t i = 0; i < allNegIndices_[n].size(); ++i) {
        confGTData_->getData()[count] = backgroundId_;
        size_t confOffset =
            n * numPriors_ * numClasses_ + allNegIndices_[n][i] * numClasses_;
        std::copy(confBufferData + confOffset,
                  confBufferData + confOffset + numClasses_,
                  confProbData + count * numClasses_);
        confPredData.reserve(confPredData.size() + numClasses_);
        confPredData.insert(confPredData.end(),
                            confBufferData + confOffset,
                            confBufferData + confOffset + numClasses_);
        ++count;
      }
    }
    CHECK_EQ(numConf_, count);
    confProb_->softmax(*confProb_);
    MatrixPtr confLossOutput;
    Matrix::resizeOrCreate(confLossOutput, numConf_, 1, false, false);
    confLossOutput->oneHotCrossEntropy(*confProb_, *confGTData_);
    confLoss_ = confLossOutput->getSum() / numConf_;
  }
  real loss = locLoss_ + confLoss_;
  MatrixPtr outV = getOutputValue();
  outV->assign(loss);
}

void RPNLossLayer::backward(const UpdateCallback& callback) {
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();
  locBuffer_->zeroMem();
  confBuffer_->zeroMem();

  // Back propagate on location prediction
  if (numMatches_ >= 1) {
    MatrixPtr locDiffBuffer;
    Matrix::resizeOrCreate(locDiffBuffer, numMatches_ * 4, 1, false, false);
    locDiffBuffer->smoothL1Bp(*locDiff_, *locGTData_, 0.0);
    locDiff_->copyFrom(*locDiffBuffer);
    // scale gradient
    for (size_t i = 0; i < numMatches_ * 4; ++i)
      locDiff_->getData()[i] *= (1. / numMatches_ * lossRatio_);
    // Copy gradient back
    size_t count = 0;
    const real* locDiffData = locDiff_->getData();
    for (size_t n = 0; n < batchSize; ++n) {
      for (size_t i = 0; i < numPriors_; ++i) {
        if (allMatchIndices_[n][i] == -1) continue;
        real* locBufferData =
            locBuffer_->getData() + n * numPriors_ * 4 + i * 4;
        std::copy(locDiffData + count * 4,
                  locDiffData + (count + 1) * 4,
                  locBufferData);
        ++count;
      }
    }
    CHECK_EQ(count, numMatches_);
  }

  if (numConf_ >= 1) {
    for (size_t i = 0; i < numConf_; ++i)
      confProb_->getData()[i * numClasses_ + confGTData_->getData()[i]] -= 1;
    for (size_t i = 0; i < numConf_ * numClasses_; ++i)
      confProb_->getData()[i] *= (1. / numConf_);
    size_t count = 0;
    const real* confProbData = confProb_->getData();
    for (size_t n = 0; n < batchSize; ++n) {
      for (size_t i = 0; i < numPriors_; ++i) {
        if (allMatchIndices_[n][i] == -1) continue;
        real* confDiffData = confBuffer_->getData() +
                             n * numPriors_ * numClasses_ + i * numClasses_;
        std::copy(confProbData + count * numClasses_,
                  confProbData + (count + 1) * numClasses_,
                  confDiffData);
        ++count;
      }
      for (size_t i = 0; i < allNegIndices_[n].size(); ++i) {
        int idx = allNegIndices_[n][i];
        real* confDiffData = confBuffer_->getData() +
                             n * numPriors_ * numClasses_ + idx * numClasses_;
        std::copy(confProbData + count * numClasses_,
                  confProbData + (count + 1) * numClasses_,
                  confDiffData);
        ++count;
      }
    }
    CHECK_EQ(count, numConf_);
  }
  if (useGpu_) {
    locTmpBuffer_->copyFrom(*locCpuBuffer_);
    confTmpBuffer_->copyFrom(*confCpuBuffer_);
    locBuffer_ = locTmpBuffer_;
    confBuffer_ = confTmpBuffer_;
  }
  // copy back
  size_t locOffset = 0;
  size_t confOffset = 0;
  auto layerConf = config_.inputs(0).rpn_loss_conf();
  for (size_t n = 0; n < inputNum_; ++n) {
    const MatrixPtr inLocG = getInputGrad(*getLocInputLayer(n));
    const MatrixPtr inConfG = getInputGrad(*getConfInputLayer(n));
    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    // only for unittest, there are no width and height information
    // when constructing matrix in unittest, so we should
    // set the shape in configuration
    if (!height) height = layerConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = layerConf.width();

    // NHWC to NCHW
    MatrixPtr locGBuffer;
    Matrix::resizeOrCreate(
        locGBuffer, inLocG->getHeight(), inLocG->getWidth(), false, useGpu_);
    MatrixPtr confGBuffer;
    Matrix::resizeOrCreate(
        confGBuffer, inConfG->getHeight(), inConfG->getWidth(), false, useGpu_);

    locOffset += decomposeWithPermute(*locBuffer_,
                                      height,
                                      width,
                                      locSizeSum_,
                                      locOffset,
                                      batchSize,
                                      *locGBuffer,
                                      kNHWCToNCHW);
    inLocG->add(*locGBuffer);
    confOffset += decomposeWithPermute(*confBuffer_,
                                       height,
                                       width,
                                       confSizeSum_,
                                       confOffset,
                                       batchSize,
                                       *confGBuffer,
                                       kNHWCToNCHW);
    inConfG->add(*confGBuffer);
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);
}

}  // namespace paddle
