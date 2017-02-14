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

#include "MultiBoxLossLayer.h"
#include <float.h>
#include <vector>
#include "DataLayer.h"
using std::vector;
using std::map;
using std::pair;

namespace paddle {

REGISTER_LAYER(multibox_loss, MultiBoxLossLayer);

bool MultiBoxLossLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  auto& mblConf = config_.inputs(0).multibox_loss_conf();
  numClasses_ = mblConf.num_classes();
  inputNum_ = mblConf.input_num();
  overlapThreshold_ = mblConf.overlap_threshold();
  negPosRatio_ = mblConf.neg_pos_ratio();
  negOverlap_ = mblConf.neg_overlap();
  backgroundId_ = mblConf.background_id();
  return true;
}

void MultiBoxLossLayer::forward(PassType passType) {
  Layer::forward(passType);
  // Same as getConfInputLayer(0)
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();
  resetOutput(batchSize, 1);

  // Allocate buffer memory
  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; n++) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    locSizeSum_ += inLoc->getElementCnt();
    confSizeSum_ += inConf->getElementCnt();
  }
  // locBuffer layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin2 ......
  // confBuffer layout:
  // | class1 score | class2 score | ... |classN score | class1 score | ......
  Matrix::resizeOrCreate(locTmpBuffer_, 1, locSizeSum_, false, useGpu_);
  Matrix::resizeOrCreate(confTmpBuffer_, 1, confSizeSum_, false, useGpu_);
  locBuffer_ = locTmpBuffer_;
  confBuffer_ = confTmpBuffer_;

  forwardDataProcess(batchSize);
  // priorValue layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var
  // | ymax1Var | xmin2 | ......
  MatrixPtr priorValue;
  // labelValue layout:
  // classX_Y means the Yth object of the Xth sample.
  // | class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ......
  // | classN_M | xminN_M | yminN_M | xmaxN_M | ymaxN_M | difficultN_M
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
  // Retrieve max scores for each prior. Used in negative mining
  vector<vector<real>> allMaxConfScore;
  numPriors_ = priorValue->getElementCnt() / 8;
  getMaxConfScore(confBuffer_->getData(),
                  batchSize,
                  numPriors_,
                  numClasses_,
                  &allMaxConfScore);

  // Match priorbox to ground truth bbox
  Argument label = getInput(*getLabelLayer());
  const int* labelIndex = label.sequenceStartPositions->getData(false);
  int seqNum = label.getNumSequences();
  numMatches_ = 0;
  numNegs_ = 0;
  allMatchIndices_.clear();
  allNegIndices_.clear();
  generateMatchIndices(
      batchSize, labelIndex, seqNum, priorValue, labelValue, allMaxConfScore);

  // BBox loc l1 smooth loss
  locLoss_ = 0;
  if (numMatches_ >= 1) {
    size_t count = 0;
    MatrixPtr locLossOutput;
    Matrix::resizeOrCreate(locLossOutput, numMatches_ * 4, 1, false, false);
    Matrix::resizeOrCreate(locGtData_, numMatches_ * 4, 1, false, false);
    Matrix::resizeOrCreate(locDiff_, numMatches_ * 4, 1, false, false);
    locDiff_->zeroMem();
    vector<real> locGtData;

    for (size_t n = 0; n < batchSize; n++) {
      for (size_t i = 0; i < numPriors_; i++) {
        if (allMatchIndices_[n][i] == -1) continue;
        size_t locOffset =
            n * (locBuffer_->getElementCnt() / batchSize) + i * 4;
        locDiff_->getData()[count++] = (locBuffer_->getData() + locOffset)[0];
        locDiff_->getData()[count++] = (locBuffer_->getData() + locOffset)[1];
        locDiff_->getData()[count++] = (locBuffer_->getData() + locOffset)[2];
        locDiff_->getData()[count++] = (locBuffer_->getData() + locOffset)[3];

        const int gtIdx = allMatchIndices_[n][i];
        vector<real> gtEncode;
        size_t priorOffset = i * 8;
        size_t labelOffset = (labelIndex[n] + gtIdx) * 6;
        encodeBBox(priorValue->getData() + priorOffset,
                   labelValue->getData() + labelOffset,
                   &gtEncode);
        locGtData.insert(locGtData.end(), gtEncode.begin(), gtEncode.end());
      }
    }
    locGtData_->copyFrom(&locGtData[0], numMatches_ * 4);
    locLossOutput->smoothL1(*locDiff_, *locGtData_);
    locLoss_ = locLossOutput->getSum() / numMatches_;
  }

  // BBox conf softmax loss
  confLoss_ = 0;
  numConf_ = numMatches_ + numNegs_;
  if (numConf_ >= 1) {
    Matrix::resizeOrCreate(confProb_, numConf_, numClasses_, false, false);
    IVector::resizeOrCreate(confGtData_, numConf_, false);
    confProb_->zeroMem();
    size_t count = 0;

    vector<real> confPredData;
    for (size_t n = 0; n < batchSize; n++) {
      for (size_t i = 0; i < numPriors_; i++) {
        if (allMatchIndices_[n][i] == -1) continue;
        size_t labelOffset = (labelIndex[n] + allMatchIndices_[n][i]) * 6;
        const int gtLabel = (labelValue->getData() + labelOffset)[0];
        confGtData_->getData()[count] = gtLabel;
        size_t confOffset = n * numPriors_ * numClasses_ + i * numClasses_;
        for (size_t j = 0; j < numClasses_; j++) {
          confProb_->getData()[count * numClasses_ + j] =
              (confBuffer_->getData() + confOffset)[j];
          confPredData.push_back((confBuffer_->getData() + confOffset)[j]);
        }
        count++;
      }
      // Negative mining samples
      for (size_t i = 0; i < allNegIndices_[n].size(); i++) {
        confGtData_->getData()[count] = backgroundId_;
        size_t confOffset =
            n * numPriors_ * numClasses_ + allNegIndices_[n][i] * numClasses_;
        for (size_t j = 0; j < numClasses_; j++) {
          confProb_->getData()[count * numClasses_ + j] =
              (confBuffer_->getData() + confOffset)[j];
          confPredData.push_back((confBuffer_->getData() + confOffset)[j]);
        }
        count++;
      }
    }
    confProb_->softmax(*confProb_);
    MatrixPtr confLossOutput;
    Matrix::resizeOrCreate(confLossOutput, numConf_, 1, false, false);
    confLossOutput->oneHotCrossEntropy(*confProb_, *confGtData_);
    confLoss_ = confLossOutput->getSum() / numMatches_;
  }
  real loss = locLoss_ + confLoss_;
  MatrixPtr outV = getOutputValue();
  vector<real> tmp(batchSize, loss);
  outV->copyFrom(&tmp[0], batchSize);
}

void MultiBoxLossLayer::backward(const UpdateCallback& callback) {
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();
  locBuffer_->zeroMem();
  confBuffer_->zeroMem();

  // Back propagate on location prediction
  if (numMatches_ >= 1) {
    locDiff_->smoothL1Bp(*locDiff_, *locGtData_);
    // Scale gradient
    for (size_t i = 0; i < numMatches_ * 4; i++)
      locDiff_->getData()[i] *= (1. / numMatches_);
    // Copy gradient back
    size_t count = 0;
    for (size_t n = 0; n < batchSize; n++)
      for (size_t i = 0; i < numPriors_; i++) {
        if (allMatchIndices_[n][i] == -1) continue;
        real* locDiffData = locBuffer_->getData() + n * numPriors_ * 4 + i * 4;
        locDiffData[0] = (locDiff_->getData() + count * 4)[0];
        locDiffData[1] = (locDiff_->getData() + count * 4)[1];
        locDiffData[2] = (locDiff_->getData() + count * 4)[2];
        locDiffData[3] = (locDiff_->getData() + count * 4)[3];
        count++;
      }
    CHECK_EQ(count, numMatches_);
  }

  // Back propagate on confidence prediction
  if (numConf_ >= 1) {
    // Caffe version softmax backpropagation
    for (size_t i = 0; i < numConf_; i++)
      confProb_->getData()[i * numClasses_ + confGtData_->getData()[i]] -= 1;
    // Scale gradient
    for (size_t i = 0; i < numConf_ * numClasses_; i++)
      confProb_->getData()[i] *= (1. / numMatches_);
    // Copy gradient back
    size_t count = 0;
    for (size_t n = 0; n < batchSize; n++) {
      for (size_t i = 0; i < numPriors_; i++) {
        if (allMatchIndices_[n][i] == -1) continue;
        real* confDiffData = confBuffer_->getData() +
                             n * numPriors_ * numClasses_ + i * numClasses_;
        for (size_t j = 0; j < numClasses_; j++)
          confDiffData[j] = (confProb_->getData() + count * numClasses_)[j];
        count++;
      }
      // Negative mining samples
      for (size_t i = 0; i < allNegIndices_[n].size(); i++) {
        int idx = allNegIndices_[n][i];
        real* confDiffData = confBuffer_->getData() +
                             n * numPriors_ * numClasses_ + idx * numClasses_;
        for (size_t j = 0; j < numClasses_; j++)
          confDiffData[j] = (confProb_->getData() + count * numClasses_)[j];
        count++;
      }
    }
    CHECK_EQ(count, numConf_);
  }
  // Copy data from CPU to GPU if use GPU
  if (useGpu_) {
    locTmpBuffer_->copyFrom(*locCpuBuffer_);
    confTmpBuffer_->copyFrom(*confCpuBuffer_);
    locBuffer_ = locTmpBuffer_;
    confBuffer_ = confTmpBuffer_;
  }
  backwardDataProcess(batchSize);
}

void MultiBoxLossLayer::generateMatchIndices(
    size_t batchSize,
    const int* labelIndex,
    int seqNum,
    MatrixPtr priorValue,
    MatrixPtr labelValue,
    vector<vector<real>> allMaxConfScore) {
  for (size_t n = 0; n < batchSize; n++) {
    vector<int> matchIndices;
    vector<int> negIndices;
    vector<real> matchOverlaps;
    matchIndices.resize(numPriors_, -1);
    matchOverlaps.resize(numPriors_, 0);
    size_t bboxNum = 0;
    if (n < (size_t)seqNum) bboxNum = labelIndex[n + 1] - labelIndex[n];
    if (!bboxNum) {
      allMatchIndices_.push_back(matchIndices);
      allNegIndices_.push_back(negIndices);
      continue;
    }
    matchBBox(priorValue->getData(),
              labelValue->getData() + labelIndex[n] * 6,
              overlapThreshold_,
              numPriors_,
              bboxNum,
              &matchIndices,
              &matchOverlaps);
    size_t numPos = 0;
    size_t numNeg = 0;
    for (size_t i = 0; i < matchIndices.size(); i++)
      if (matchIndices[i] != -1) numPos++;
    numMatches_ += numPos;
    // Negtive mining
    vector<pair<real, size_t>> scoresIndices;
    for (size_t i = 0; i < matchIndices.size(); i++)
      if (matchIndices[i] == -1 && matchOverlaps[i] < negOverlap_) {
        scoresIndices.push_back(std::make_pair(allMaxConfScore[n][i], i));
        numNeg++;
      }
    // Pick top num_neg negatives
    numNeg = std::min(static_cast<size_t>(numPos * negPosRatio_), numNeg);
    std::sort(scoresIndices.begin(), scoresIndices.end(), sortScorePairDescend);
    for (size_t i = 0; i < numNeg; i++)
      negIndices.push_back(scoresIndices[i].second);
    numNegs_ += numNeg;
    allMatchIndices_.push_back(matchIndices);
    allNegIndices_.push_back(negIndices);
  }
}

void MultiBoxLossLayer::forwardDataProcess(size_t batchSize) {
  // BatchTrans (from NCHW to NHWC) and concat
  size_t locOffset = 0;
  size_t confOffset = 0;
  // For unit test
  auto& mblConf = config_.inputs(0).multibox_loss_conf();
  // Each layer input has different size
  for (size_t n = 0; n < inputNum_; n++) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));

    size_t locSize = inLoc->getElementCnt();
    size_t confSize = inConf->getElementCnt();
    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    if (!height) height = mblConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = mblConf.width();
    size_t locChannels = locSize / (height * width * batchSize);
    size_t confChannels = confSize / (height * width * batchSize);
    size_t imgPixels = height * width;

    for (size_t i = 0; i < batchSize; i++) {
      // Concat with axis N (NCHW -> NHWC)
      size_t locBatchOffset = i * (locSizeSum_ / batchSize) + locOffset;
      size_t confBatchOffset = i * (confSizeSum_ / batchSize) + confOffset;
      const MatrixPtr inLocTmp =
          Matrix::create(inLoc->getData() + i * locChannels * imgPixels,
                         locChannels,
                         imgPixels,
                         false,
                         useGpu_);
      MatrixPtr outLocTmp =
          Matrix::create(locBuffer_->getData() + locBatchOffset,
                         imgPixels,
                         locChannels,
                         false,
                         useGpu_);
      inLocTmp->transpose(outLocTmp, false);
      const MatrixPtr inConfTmp =
          Matrix::create(inConf->getData() + i * confChannels * imgPixels,
                         confChannels,
                         imgPixels,
                         false,
                         useGpu_);
      MatrixPtr outConfTmp =
          Matrix::create(confBuffer_->getData() + confBatchOffset,
                         imgPixels,
                         confChannels,
                         false,
                         useGpu_);
      inConfTmp->transpose(outConfTmp, false);
    }
    locOffset += locChannels * imgPixels;
    confOffset += confChannels * imgPixels;
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);
}

void MultiBoxLossLayer::backwardDataProcess(size_t batchSize) {
  // Copy grad back to input
  size_t locOffset = 0;
  size_t confOffset = 0;
  // Each input has different size
  auto mblConf = config_.inputs(0).multibox_loss_conf();
  for (size_t n = 0; n < inputNum_; n++) {
    const MatrixPtr inLocG = getInputGrad(*getLocInputLayer(n));
    const MatrixPtr inConfG = getInputGrad(*getConfInputLayer(n));
    size_t locSize = inLocG->getElementCnt();
    size_t confSize = inConfG->getElementCnt();
    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    if (!height) height = mblConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = mblConf.width();
    size_t locChannels = locSize / (height * width * batchSize);
    size_t confChannels = confSize / (height * width * batchSize);
    size_t imgPixels = height * width;

    for (size_t i = 0; i < batchSize; i++) {
      size_t locBatchOffset = i * (locSizeSum_ / batchSize) + locOffset;
      size_t confBatchOffset = i * (confSizeSum_ / batchSize) + confOffset;
      const MatrixPtr inLocTmp =
          Matrix::create(locBuffer_->getData() + locBatchOffset,
                         imgPixels,
                         locChannels,
                         false,
                         useGpu_);
      MatrixPtr outLocTmp =
          Matrix::create(inLocG->getData() + i * locChannels * imgPixels,
                         locChannels,
                         imgPixels,
                         false,
                         useGpu_);
      inLocTmp->transpose(outLocTmp, false);
      const MatrixPtr inConfTmp =
          Matrix::create(confBuffer_->getData() + confBatchOffset,
                         imgPixels,
                         confChannels,
                         false,
                         useGpu_);
      MatrixPtr outConfTmp =
          Matrix::create(inConfG->getData() + i * confChannels * imgPixels,
                         confChannels,
                         imgPixels,
                         false,
                         useGpu_);
      inConfTmp->transpose(outConfTmp, false);
    }
    locOffset += locChannels * imgPixels;
    confOffset += confChannels * imgPixels;
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);
}

void MultiBoxLossLayer::getMaxConfScore(const real* confData,
                                        const size_t batchSize,
                                        const size_t numPriors,
                                        const size_t numClasses,
                                        vector<vector<real>>* allMaxConfScore) {
  allMaxConfScore->clear();
  for (size_t i = 0; i < batchSize; i++) {
    vector<real> maxConfScore;
    for (size_t j = 0; j < numPriors; j++) {
      int offset = j * numClasses;
      real maxVal = -FLT_MAX;
      real maxPosVal = -FLT_MAX;
      real maxScore = 0.;
      for (size_t c = 0; c < numClasses; c++) {
        maxVal = std::max<real>(confData[offset + c], maxVal);
        if (c != backgroundId_)
          maxPosVal = std::max<real>(confData[offset + c], maxPosVal);
      }
      real sum = 0;
      for (size_t c = 0; c < numClasses; c++)
        sum += std::exp(confData[offset + c] - maxVal);
      maxScore = std::exp(maxPosVal - maxVal) / sum;
      maxConfScore.push_back(maxScore);
    }
    confData += numPriors * numClasses;
    allMaxConfScore->push_back(maxConfScore);
  }
}

real MultiBoxLossLayer::jaccardOverlap(const real* priorData,
                                       const real* labelData) {
  real xMin1 = priorData[0];
  real yMin1 = priorData[1];
  real xMax1 = priorData[2];
  real yMax1 = priorData[3];
  real xMin2 = labelData[1];
  real yMin2 = labelData[2];
  real xMax2 = labelData[3];
  real yMax2 = labelData[4];

  real width1 = xMax1 - xMin1;
  real height1 = yMax1 - yMin1;
  real width2 = xMax2 - xMin2;
  real height2 = yMax2 - yMin2;
  real bboxSize1 = width1 * height1;
  real bboxSize2 = width2 * height2;

  real intersectWidth;
  real intersectHeight;

  if (!(xMin1 > xMax2 || yMin1 > yMax2 || xMax1 < xMin2 || yMax1 < yMin2)) {
    intersectWidth = std::min(xMax1, xMax2) - std::max(xMin1, xMin2);
    intersectHeight = std::min(yMax1, yMax2) - std::max(yMin1, yMin2);
    real intersectSize = intersectWidth * intersectHeight;
    real overlap = intersectSize / (bboxSize1 + bboxSize2 - intersectSize);
    return overlap;
  } else {
    return 0;
  }
}

void MultiBoxLossLayer::matchBBox(const real* priorData,
                                  const real* labelData,
                                  const real overlapThreshold,
                                  const size_t numPriors,
                                  const size_t bboxNum,
                                  vector<int>* matchIndices,
                                  vector<real>* matchOverlaps) {
  map<size_t, map<size_t, real>> overlaps;
  for (size_t i = 0; i < numPriors; i++)
    for (size_t j = 0; j < bboxNum; j++) {
      real overlap = jaccardOverlap(priorData + i * 8, labelData + j * 6);
      if (overlap > 1e-6) {
        (*matchOverlaps)[i] = std::max(overlap, (*matchOverlaps)[i]);
        overlaps[i][j] = overlap;
      }
    }
  // Bipartite matching
  vector<bool> gtMask;
  gtMask.resize(bboxNum, true);
  size_t bboxCount = bboxNum;
  while (bboxCount) {
    int maxPriorIdx = -1;
    int maxGtIdx = -1;
    real maxOverlap = -1;
    for (map<size_t, map<size_t, real>>::iterator iter = overlaps.begin();
         iter != overlaps.end();
         iter++) {
      size_t priorIdx = iter->first;
      if ((*matchIndices)[priorIdx] != -1) continue;
      for (size_t gtIdx = 0; gtIdx < bboxNum; gtIdx++)
        if (gtMask[gtIdx] && iter->second.find(gtIdx) != iter->second.end() &&
            iter->second[gtIdx] > maxOverlap) {
          maxPriorIdx = (int)priorIdx;
          maxGtIdx = (int)gtIdx;
          maxOverlap = iter->second[gtIdx];
        }
    }
    if (maxPriorIdx == -1) {
      break;
    } else {
      (*matchIndices)[maxPriorIdx] = maxGtIdx;
      (*matchOverlaps)[maxPriorIdx] = maxOverlap;
      gtMask[maxGtIdx] = false;
      bboxCount--;
    }
  }
  // Per prediction match
  for (map<size_t, map<size_t, real>>::iterator iter = overlaps.begin();
       iter != overlaps.end();
       iter++) {
    size_t priorIdx = iter->first;
    if ((*matchIndices)[priorIdx] != -1) continue;
    int maxGtIdx = -1;
    real maxOverlap = -1;
    for (size_t gtIdx = 0; gtIdx < bboxNum; gtIdx++)
      if (iter->second.find(gtIdx) != iter->second.end()) {
        real overlap = iter->second[gtIdx];
        if (overlap > maxOverlap && overlap > overlapThreshold) {
          maxGtIdx = gtIdx;
          maxOverlap = overlap;
        }
      }
    if (maxGtIdx != -1) {
      (*matchIndices)[priorIdx] = maxGtIdx;
      (*matchOverlaps)[priorIdx] = maxOverlap;
    }
  }
}

void MultiBoxLossLayer::encodeBBox(const real* priorData,
                                   const real* labelData,
                                   vector<real>* gtEncode) {
  real priorWidth = priorData[2] - priorData[0];
  real priorHeight = priorData[3] - priorData[1];
  real priorCenterX = (priorData[0] + priorData[2]) / 2;
  real priorCenterY = (priorData[1] + priorData[3]) / 2;

  real gtWidth = labelData[3] - labelData[1];
  real gtHeight = labelData[4] - labelData[2];
  real gtCenterX = (labelData[1] + labelData[3]) / 2;
  real gtCenterY = (labelData[2] + labelData[4]) / 2;

  gtEncode->push_back((gtCenterX - priorCenterX) / priorWidth / priorData[4]);
  gtEncode->push_back((gtCenterY - priorCenterY) / priorHeight / priorData[5]);
  gtEncode->push_back(std::log(gtWidth / priorWidth) / priorData[6]);
  gtEncode->push_back(std::log(gtHeight / priorHeight) / priorData[7]);
}
}  // namespace paddle
