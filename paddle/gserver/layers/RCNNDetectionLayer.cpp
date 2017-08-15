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

#include "RCNNDetectionLayer.h"

namespace paddle {

REGISTER_LAYER(rcnn_detection, RCNNDetectionLayer);

bool RCNNDetectionLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto& layerConf = config_.inputs(0).rcnn_detection_conf();
  nmsThreshold_ = layerConf.nms_threshold();
  confidenceThreshold_ = layerConf.confidence_threshold();
  nmsTopK_ = layerConf.nms_top_k();
  keepTopK_ = layerConf.keep_top_k();
  numClasses_ = layerConf.num_classes();
  backgroundId_ = layerConf.background_id();
  return true;
}

void RCNNDetectionLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr roiValue = getInputValue(0);
  MatrixPtr locPredValue = getInputValue(1);
  MatrixPtr confPredValue = getInputValue(2);

  // do softmax
  MatrixPtr confPredNormValue;
  Matrix::resizeOrCreate(confPredNormValue,
                         confPredValue->getHeight(),
                         confPredValue->getWidth(),
                         false,
                         useGpu_);
  confPredNormValue->copyFrom(*confPredValue);
  confPredNormValue->softmax(*confPredNormValue);
  confPredValue = confPredNormValue;

  if (useGpu_) {  // copy data from GPU
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

  // The format of the RoI is:
  // | batch_idx | xmin | ymin | xmax | ymax |
  real* roisData = roiValue->getData();
  size_t roiDim = roiValue->getWidth();
  size_t roiNum = roiValue->getHeight();
  real* locPredData = locPredValue->getData();
  real* confPredData = confPredValue->getData();

  // <batchIdx, <classIdx, <(score, box)>>>
  std::map<size_t,
           std::map<size_t, std::vector<std::pair<real, NormalizedBBox>>>>
      allDecodedBBoxes;
  for (size_t n = 0; n < roiNum; ++n) {
    int batchIdx = *(roisData + n * roiDim);
    std::vector<real> roiLocData(4);  // RoI location
    for (size_t j = 0; j < 4; ++j) {
      roiLocData[j] = *(roisData + n * roiDim + 1 + j);
    }
    // location predictions for each class
    for (size_t c = 0; c < numClasses_; ++c) {
      if (c == backgroundId_) continue;
      std::vector<real> predLocData(4);
      for (size_t j = 0; j < 4; ++j) {
        predLocData[j] = *(locPredData + n * numClasses_ * 4 + c * 4 + j);
      }
      real predConfData = *(confPredData + n * numClasses_ + c);
      allDecodedBBoxes[batchIdx][c].push_back(
          std::make_pair(predConfData, decodeBBox(roiLocData, predLocData)));
    }
  }
  // <batchIdx, <classIdx, <bboxIdxes>>
  std::map<size_t, std::map<size_t, std::vector<size_t>>> allIndices;
  size_t numKept = getDetectionIndices(backgroundId_,
                                       confidenceThreshold_,
                                       nmsTopK_,
                                       nmsThreshold_,
                                       keepTopK_,
                                       allDecodedBBoxes,
                                       &allIndices);
  resetOutput(numKept, 7);
  MatrixPtr outV = getOutputValue();
  getDetectionOutput(numKept, allIndices, allDecodedBBoxes, *outV);
}

}  // namespace paddle
