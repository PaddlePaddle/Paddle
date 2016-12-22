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

#include <memory>
#include <vector>
#include "Layer.h"

using std::vector;

namespace paddle {

/**
 * The multibox loss layer for a SSD detection task.
 * The loss is composed by the location loss and the confidence loss.
 * The location loss is a smooth L1 loss and the confidence loss is
 * a softmax loss.
 * Reference:
 *    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
 *    Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector
 */

class MultiBoxLossLayer : public Layer {
public:
  explicit MultiBoxLossLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getPriorBoxLayer() { return inputLayers_[0]; }
  LayerPtr getLabelLayer() { return inputLayers_[1]; }
  LayerPtr getLocInputLayer(size_t index) { return inputLayers_[2 + index]; }
  LayerPtr getConfInputLayer(size_t index) {
    return inputLayers_[2 + inputNum_ + index];
  }

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

  real jaccardOverlap(const real* priorData, const real* labelData);

  void getMaxConfScore(const real* confData,
                       const size_t batchSize,
                       const size_t numPriors,
                       const size_t numClasses,
                       vector<vector<real>>* allMaxConfScore);

  void matchBBox(const real* priorData,
                 const real* labelData,
                 const real overlapThreshold,
                 const size_t numPriors,
                 const size_t bboxNum,
                 vector<int>* matchIndices,
                 vector<real>* matchOverlaps);

  void encodeBBox(const real* priorData,
                  const real* labelData,
                  vector<real>* gtEncode);

  real smoothL1Loss(const vector<real> locPredData,
                    const vector<real> locGtData,
                    const real locWeight,
                    real* locDiff);

  void smoothL1LossBp(const real locLoss,
                      const size_t numMatches,
                      real* locDiff);

  real softmaxLoss(const vector<real> confPredData,
                   const vector<int> confGtData,
                   const size_t numClasses,
                   const size_t numMatches,
                   real* confProb);

  void softmaxLossBp(const vector<int> confGtData,
                     const size_t numClasses,
                     real* confProb);

protected:
  size_t numClasses_ = 21;
  real locWeight_ = 1.0;
  real overlapThreshold_ = 0.5;
  real negPosRatio_ = 3.0;
  real negOverlap_ = 0.5;
  size_t inputNum_ = 1;
  size_t backgroundId_ = 0;

  real locLoss_;
  real confLoss_;

  size_t numPriors_;
  size_t numMatches_;
  size_t numNegs_;
  size_t numConf_;
  size_t locSizeSum_;
  size_t confSizeSum_;

  vector<vector<int>> allMatchIndices_;
  vector<vector<int>> allNegIndices_;
  vector<int> confGtData_;

  // Temporary buffers for permute and flat mbox loc and conf input
  MatrixPtr locBuffer_;
  MatrixPtr confBuffer_;
  MatrixPtr locDiff_;
  MatrixPtr confProb_;

  MatrixPtr locCpuBuffer_;
  MatrixPtr confCpuBuffer_;
  MatrixPtr locGpuBuffer_;
  MatrixPtr confGpuBuffer_;
  MatrixPtr priorCpuValue_;
  MatrixPtr labelCpuValue_;
};

}  // namespace paddle
