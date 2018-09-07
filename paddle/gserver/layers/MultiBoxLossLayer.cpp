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

#include "MultiBoxLossLayer.h"
#include <float.h>
#include <vector>
#include "DataLayer.h"

namespace paddle {

REGISTER_LAYER(multibox_loss, MultiBoxLossLayer);

bool MultiBoxLossLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  auto layerConf = config_.inputs(0).multibox_loss_conf();
  numClasses_ = layerConf.num_classes();
  inputNum_ = layerConf.input_num();
  overlapThreshold_ = layerConf.overlap_threshold();
  negPosRatio_ = layerConf.neg_pos_ratio();
  negOverlap_ = layerConf.neg_overlap();
  backgroundId_ = layerConf.background_id();
  return true;
}

void MultiBoxLossLayer::forward(PassType passType) {
  Layer::forward(passType);
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();
  resetOutput(batchSize, 1);

  // all location data and confidence score data
  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; ++n) {
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
  auto& layerConf = config_.inputs(0).multibox_loss_conf();
  for (size_t n = 0; n < inputNum_; ++n) {
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
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var | ymax1Var
  // | xmin2 | ......
  MatrixPtr priorValue;

  // labelValue layout:
  // | class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ......
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

  // Get max scores for each prior bbox. Used in negative mining
  std::vector<std::vector<real>> allMaxConfScore;
  numPriors_ = priorValue->getElementCnt() / 8;
  getMaxConfidenceScores(confBuffer_->getData(),
                         batchSize,
                         numPriors_,
                         numClasses_,
                         backgroundId_,
                         &allMaxConfScore);

  // Match prior bbox to groundtruth bbox
  Argument label = getInput(*getLabelLayer());
  const int* labelIndex = label.sequenceStartPositions->getData(false);
  size_t seqNum = label.getNumSequences();
  numMatches_ = 0;
  numNegs_ = 0;
  allMatchIndices_.clear();
  allNegIndices_.clear();

  std::pair<size_t, size_t> retPair = generateMatchIndices(*priorValue,
                                                           numPriors_,
                                                           *labelValue,
                                                           labelIndex,
                                                           seqNum,
                                                           allMaxConfScore,
                                                           batchSize,
                                                           overlapThreshold_,
                                                           negOverlap_,
                                                           negPosRatio_,
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
        size_t priorOffset = i * 8;
        std::vector<NormalizedBBox> priorBBoxVec;
        getBBoxFromPriorData(
            priorValue->getData() + priorOffset, 1, priorBBoxVec);
        std::vector<std::vector<real>> priorBBoxVar;
        getBBoxVarFromPriorData(
            priorValue->getData() + priorOffset, 1, priorBBoxVar);
        size_t labelOffset = (labelIndex[n] + gtIdx) * 6;
        std::vector<NormalizedBBox> gtBBoxVec;
        getBBoxFromLabelData(labelValue->getData() + labelOffset, 1, gtBBoxVec);
        std::vector<real> gtEncode;
        encodeBBoxWithVar(
            priorBBoxVec[0], priorBBoxVar[0], gtBBoxVec[0], gtEncode);
        locGTData.insert(locGTData.end(), gtEncode.begin(), gtEncode.end());
      }
    }
    locGTData_->copyFrom(&locGTData[0], numMatches_ * 4);
    locLossOutput->smoothL1(*locDiff_, *locGTData_, 0.0);
    locLoss_ = locLossOutput->getSum() / numMatches_;
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
        size_t labelOffset = (labelIndex[n] + allMatchIndices_[n][i]) * 6;
        const int gtLabel = (labelValue->getData() + labelOffset)[0];
        confGTData_->getData()[count] = gtLabel;
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
      // Negative mining samples
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
    confLoss_ = confLossOutput->getSum() / numMatches_;
  }
  real loss = locLoss_ + confLoss_;
  MatrixPtr outV = getOutputValue();
  outV->assign(loss);
}

void MultiBoxLossLayer::backward(const UpdateCallback& callback) {
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
      locDiff_->getData()[i] *= (1. / numMatches_);
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
      confProb_->getData()[i] *= (1. / numMatches_);
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
  auto layerConf = config_.inputs(0).multibox_loss_conf();
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
