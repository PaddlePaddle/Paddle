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

#include "RCNNLossLayer.h"
#include <vector>

namespace paddle {

REGISTER_LAYER(rcnn_loss, RCNNLossLayer);

bool RCNNLossLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto layerConf = config_.inputs(0).rcnn_loss_conf();
  lossRatio_ = layerConf.loss_ratio();
  numClasses_ = layerConf.num_classes();
  backgroundId_ = layerConf.background_id();
  return true;
}

void RCNNLossLayer::forward(PassType passType) {
  Layer::forward(passType);
  // the format of each RoI from ProposalTargetLayer:
  // | batchIdx | xmin | ymin | xmax | ymax | class1 | target1 | target2 |
  // target3 | target4 |
  MatrixPtr roiValue = getInputValue(0);
  // numROIs * numClasses * 4
  MatrixPtr locPredValue = getInputValue(1);
  // numROIs * numClasses
  MatrixPtr confPredValue = getInputValue(2);

  if (useGpu_) {
    MatrixPtr roiCpuBuffer;
    Matrix::resizeOrCreate(roiCpuBuffer,
                           roiValue->getHeight(),
                           roiValue->getWidth(),
                           false,
                           false);
    MatrixPtr locCpuBuffer;
    Matrix::resizeOrCreate(locCpuBuffer,
                           locPredValue->getHeight(),
                           locPredValue->getWidth(),
                           false,
                           false);
    MatrixPtr confCpuBuffer;
    Matrix::resizeOrCreate(confCpuBuffer,
                           confPredValue->getHeight(),
                           confPredValue->getWidth(),
                           false,
                           false);
    roiCpuBuffer->copyFrom(*roiValue);
    locCpuBuffer->copyFrom(*locPredValue);
    confCpuBuffer->copyFrom(*confPredValue);
    roiValue = roiCpuBuffer;
    locPredValue = locCpuBuffer;
    confPredValue = confCpuBuffer;
  }

  real* roisData = roiValue->getData();
  size_t roiDim = roiValue->getWidth();
  size_t batchSize = roiValue->getHeight();
  CHECK_EQ(batchSize * numClasses_ * 4, locPredValue->getElementCnt());
  CHECK_EQ(batchSize * numClasses_, confPredValue->getElementCnt());
  posROIs_.clear();
  for (size_t n = 0; n < batchSize; ++n) {
    size_t classId = *(roisData + n * roiDim + 5);
    if (classId != backgroundId_) {
      posROIs_.push_back(std::make_pair(n, classId));
    }
  }

  real locLoss = 0., confLoss = 0.;

  // smooth L1 loss for BBox location, only for Positive ROIs
  if (posROIs_.size() > 0) {
    MatrixPtr locLossOutput;
    Matrix::resizeOrCreate(locLossOutput, posROIs_.size() * 4, 1, false, false);
    Matrix::resizeOrCreate(locPosGT_, posROIs_.size() * 4, 1, false, false);
    Matrix::resizeOrCreate(locPosPred_, posROIs_.size() * 4, 1, false, false);
    real* locPosGTData = locPosGT_->getData();
    real* locPosPredData = locPosPred_->getData();
    real* locPredData = locPredValue->getData();
    for (size_t n = 0; n < posROIs_.size(); ++n) {
      size_t roiIdx = posROIs_[n].first;
      size_t roiClassId = posROIs_[n].second;
      size_t locGTOffset = roiIdx * roiDim + 6;
      std::copy(roisData + locGTOffset,
                roisData + locGTOffset + 4,
                locPosGTData + n * 4);
      size_t locPredOffset =
          roiIdx * (locPredValue->getElementCnt() / batchSize) + roiClassId * 4;
      std::copy(locPredData + locPredOffset,
                locPredData + locPredOffset + 4,
                locPosPredData + n * 4);
    }
    locLossOutput->smoothL1(*locPosPred_, *locPosGT_, 0.0);
    locLoss = locLossOutput->getSum() / posROIs_.size() * lossRatio_;
  }

  // softmax loss for BBox classification confidence
  MatrixPtr confLossOutput;
  Matrix::resizeOrCreate(confLossOutput, batchSize, 1, false, false);
  IVector::resizeOrCreate(confGT_, batchSize, false);
  Matrix::resizeOrCreate(confPred_,
                         confPredValue->getHeight(),
                         confPredValue->getWidth(),
                         false,
                         false);
  confPred_->copyFrom(*confPredValue);
  auto* confGTData = confGT_->getData();
  for (size_t n = 0; n < batchSize; ++n) {
    size_t confGTOffset = n * roiDim + 5;
    confGTData[n] = *(roisData + confGTOffset);
  }
  confPred_->softmax(*confPred_);
  confLossOutput->oneHotCrossEntropy(*confPred_, *confGT_);
  confLoss = confLossOutput->getSum() / batchSize;

  real loss = locLoss + confLoss;
  resetOutput(1, 1);
  MatrixPtr outV = getOutputValue();
  outV->assign(loss);
}

void RCNNLossLayer::backward(const UpdateCallback& callback) {
  size_t batchSize = getInputValue(0)->getHeight();
  MatrixPtr inLocGrad = getInputGrad(1);
  MatrixPtr inConfGrad = getInputGrad(2);

  if (useGpu_) {
    MatrixPtr locGradCpuBuffer;
    Matrix::resizeOrCreate(locGradCpuBuffer,
                           inLocGrad->getHeight(),
                           inLocGrad->getWidth(),
                           false,
                           false);
    MatrixPtr confGradCpuBuffer;
    Matrix::resizeOrCreate(confGradCpuBuffer,
                           inConfGrad->getHeight(),
                           inConfGrad->getWidth(),
                           false,
                           false);
    locGradCpuBuffer->copyFrom(*inLocGrad);
    confGradCpuBuffer->copyFrom(*inConfGrad);
    inLocGrad = locGradCpuBuffer;
    inConfGrad = confGradCpuBuffer;
  }

  // Back propagation on location prediction
  if (posROIs_.size() > 0) {
    MatrixPtr inLocGradBuf;
    Matrix::resizeOrCreate(inLocGradBuf,
                           inLocGrad->getHeight(),
                           inLocGrad->getWidth(),
                           false,
                           false);
    inLocGradBuf->zeroMem();
    real* inLocGradBufData = inLocGradBuf->getData();
    MatrixPtr locPosGrad;
    Matrix::resizeOrCreate(locPosGrad, posROIs_.size() * 4, 1, false, false);
    locPosGrad->smoothL1Bp(*locPosPred_, *locPosGT_, 0.0);
    real* locPosGradData = locPosGrad->getData();
    // scale gradient
    for (size_t n = 0; n < posROIs_.size() * 4; ++n)
      locPosGradData[n] *= (1. / posROIs_.size() * lossRatio_);
    // Copy gradient back
    for (size_t n = 0; n < posROIs_.size(); ++n) {
      size_t roiIdx = posROIs_[n].first;
      size_t roiClassId = posROIs_[n].second;
      size_t locGradOffset =
          roiIdx * (inLocGrad->getElementCnt() / batchSize) + roiClassId * 4;
      std::copy(locPosGradData + n * 4,
                locPosGradData + (n + 1) * 4,
                inLocGradBufData + locGradOffset);
    }
    inLocGrad->add(*inLocGradBuf);
  }

  // Back propagation on classificaton prediction
  for (size_t n = 0; n < batchSize; ++n)
    confPred_->getData()[n * numClasses_ + confGT_->getData()[n]] -= 1;
  for (size_t n = 0; n < batchSize * numClasses_; ++n)
    confPred_->getData()[n] *= (1. / batchSize);
  inConfGrad->add(*confPred_);

  if (useGpu_) {
    getInputGrad(1)->copyFrom(*inLocGrad);
    getInputGrad(2)->copyFrom(*inConfGrad);
  }
}

}  // namespace paddle
