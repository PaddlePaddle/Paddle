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

#include <algorithm>
#include <map>
#include <vector>
#include "DetectionUtil.h"
#include "Layer.h"

using std::vector;
using std::pair;
using std::map;

namespace paddle {

/**
 * The detection output layer to generate proposals in RPN of Faster R-CNN.
 * This layer applies Non-maximum suppression to the all predicted bounding
 * box and keeps the Top-K bounding boxes.
 * - Input: This layer needs three input layers: The first input layer
 *          is the anchor layer. The rest two input layers are convolution
 *          layers for generating bbox location offset and the classification
 *          confidence.
 * - Output: The predict bounding box locations.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 */

class ProposalLayer : public Layer {
public:
  explicit ProposalLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr) {}

protected:
  inline LayerPtr getPriorBoxLayer() { return inputLayers_[0]; }

  inline LayerPtr getLocInputLayer(size_t index) {
    return inputLayers_[1 + index];
  }

  inline LayerPtr getConfInputLayer(size_t index) {
    return inputLayers_[1 + inputNum_ + index];
  }

  struct UnnormalizedBBox : BBoxBase<real> {
    UnnormalizedBBox() : BBoxBase<real>() {}
    real getWidth() const { return xMax - xMin + 1; }
    real getHeight() const { return yMax - yMin + 1; }
  };

  real jaccardOverlap(const UnnormalizedBBox& bbox1,
                      const UnnormalizedBBox& bbox2);

  void applyNMSFast(const vector<UnnormalizedBBox>& bboxes,
                    const real* confScoreData,
                    size_t classIdx,
                    size_t topK,
                    real confThreshold,
                    real nmsThreshold,
                    real minWidth,
                    real minHeight,
                    size_t numPriorBBoxes,
                    size_t numClasses,
                    vector<size_t>* indices);

  size_t getDetectionIndices(
      const real* confData,
      const size_t numPriorBBoxes,
      const size_t numClasses,
      const size_t backgroundId,
      const size_t batchSize,
      const size_t confThreshold,
      const size_t nmsTopK,
      const real nmsThreshold,
      const size_t keepTopK,
      const real minWidth,
      const real minHeight,
      const vector<vector<UnnormalizedBBox>>& allDecodedBBoxes,
      vector<map<size_t, vector<size_t>>>* allDetectionIndices);

  void getDetectionOutput(
      const real* confData,
      const size_t numKept,
      const size_t numPriorBBoxes,
      const size_t numClasses,
      const size_t batchSize,
      const vector<map<size_t, vector<size_t>>>& allIndices,
      const vector<vector<UnnormalizedBBox>>& allDecodedBBoxes,
      Matrix& out);

  void decodeTarget(const std::vector<real>& anchorBoxData,
                    const std::vector<real>& locPredData,
                    UnnormalizedBBox& predBox);

private:
  real nmsThreshold_;
  real confidenceThreshold_;
  size_t nmsTopK_;
  size_t keepTopK_;
  real minWidth_;
  real minHeight_;

  size_t numClasses_;
  size_t inputNum_;
  size_t backgroundId_;

  size_t locSizeSum_;
  size_t confSizeSum_;

  MatrixPtr locBuffer_;
  MatrixPtr confBuffer_;
  MatrixPtr locTmpBuffer_;
  MatrixPtr confTmpBuffer_;
  MatrixPtr priorCpuValue_;
  MatrixPtr locCpuBuffer_;
  MatrixPtr confCpuBuffer_;
};

}  // namespace paddle
