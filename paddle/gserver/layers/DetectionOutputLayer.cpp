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

#include "DetectionOutputLayer.h"

namespace paddle {

REGISTER_LAYER(detection_output, DetectionOutputLayer);

bool DetectionOutputLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto& layerConf = config_.inputs(0).detection_output_conf();
  numClasses_ = layerConf.num_classes();
  inputNum_ = layerConf.input_num();
  nmsThreshold_ = layerConf.nms_threshold();
  confidenceThreshold_ = layerConf.confidence_threshold();
  nmsTopK_ = layerConf.nms_top_k();
  keepTopK_ = layerConf.keep_top_k();
  backgroundId_ = layerConf.background_id();
  return true;
}

void DetectionOutputLayer::forward(PassType passType) {
  Layer::forward(passType);
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();

  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; ++n) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    locSizeSum_ += inLoc->getElementCnt();
    confSizeSum_ += inConf->getElementCnt();
  }

  Matrix::resizeOrCreate(locTmpBuffer_, 1, locSizeSum_, false, useGpu_);
  Matrix::resizeOrCreate(
      confTmpBuffer_, confSizeSum_ / numClasses_, numClasses_, false, useGpu_);

  size_t locOffset = 0;
  size_t confOffset = 0;
  auto& layerConf = config_.inputs(0).detection_output_conf();
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
                                   *locTmpBuffer_,
                                   kNCHWToNHWC);
    confOffset += appendWithPermute(*inConf,
                                    height,
                                    width,
                                    confSizeSum_,
                                    confOffset,
                                    batchSize,
                                    *confTmpBuffer_,
                                    kNCHWToNHWC);
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);

  MatrixPtr priorValue;
  if (useGpu_) {
    Matrix::resizeOrCreate(locCpuBuffer_, 1, locSizeSum_, false, false);
    Matrix::resizeOrCreate(
        confCpuBuffer_, confSizeSum_ / numClasses_, numClasses_, false, false);
    MatrixPtr priorTmpValue = getInputValue(*getPriorBoxLayer());
    Matrix::resizeOrCreate(
        priorCpuValue_, 1, priorTmpValue->getElementCnt(), false, false);

    locCpuBuffer_->copyFrom(*locTmpBuffer_);
    confCpuBuffer_->copyFrom(*confTmpBuffer_);
    priorCpuValue_->copyFrom(*priorTmpValue);

    locBuffer_ = locCpuBuffer_;
    confBuffer_ = confCpuBuffer_;
    priorValue = priorCpuValue_;
  } else {
    priorValue = getInputValue(*getPriorBoxLayer());
    locBuffer_ = locTmpBuffer_;
    confBuffer_ = confTmpBuffer_;
  }
  confBuffer_->softmax(*confBuffer_);

  size_t numPriors = priorValue->getElementCnt() / 8;
  std::vector<std::vector<NormalizedBBox>> allDecodedBBoxes;
  for (size_t n = 0; n < batchSize; ++n) {
    std::vector<NormalizedBBox> decodedBBoxes;
    for (size_t i = 0; i < numPriors; ++i) {
      size_t priorOffset = i * 8;
      size_t locPredOffset = n * numPriors * 4 + i * 4;
      std::vector<NormalizedBBox> priorBBoxVec;
      getBBoxFromPriorData(
          priorValue->getData() + priorOffset, 1, priorBBoxVec);
      std::vector<std::vector<real>> priorBBoxVar;
      getBBoxVarFromPriorData(
          priorValue->getData() + priorOffset, 1, priorBBoxVar);
      std::vector<real> locPredData;
      for (size_t j = 0; j < 4; ++j)
        locPredData.push_back(*(locBuffer_->getData() + locPredOffset + j));
      NormalizedBBox bbox =
          decodeBBoxWithVar(priorBBoxVec[0], priorBBoxVar[0], locPredData);
      decodedBBoxes.push_back(bbox);
    }
    allDecodedBBoxes.push_back(decodedBBoxes);
  }

  std::vector<std::map<size_t, std::vector<size_t>>> allIndices;
  size_t numKept = getDetectionIndices(confBuffer_->getData(),
                                       numPriors,
                                       numClasses_,
                                       backgroundId_,
                                       batchSize,
                                       confidenceThreshold_,
                                       nmsTopK_,
                                       nmsThreshold_,
                                       keepTopK_,
                                       allDecodedBBoxes,
                                       &allIndices);

  if (numKept > 0) {
    resetOutput(numKept, 7);
  } else {
    MatrixPtr outV = getOutputValue();
    if (outV) outV->resize(0, 0);
    return;
  }
  MatrixPtr outV = getOutputValue();
  getDetectionOutput(confBuffer_->getData(),
                     numKept,
                     numPriors,
                     numClasses_,
                     batchSize,
                     allIndices,
                     allDecodedBBoxes,
                     *outV);
}

}  // namespace paddle
