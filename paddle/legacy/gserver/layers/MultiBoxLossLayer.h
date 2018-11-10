/* copyright (c) 2016 paddlepaddle authors. all rights reserve.

licensed under the apache license, version 2.0 (the "license");
you may not use this file except in compliance with the license.
you may obtain a copy of the license at

    http://www.apache.org/licenses/license-2.0

unless required by applicable law or agreed to in writing, software
distributed under the license is distributed on an "as is" basis,
without warranties or conditions of any kind, either express or implied.
see the license for the specific language governing permissions and
limitations under the license. */

#pragma once

#include <vector>
#include "CostLayer.h"
#include "DataLayer.h"
#include "DetectionUtil.h"
#include "Layer.h"

using std::vector;
using std::pair;

namespace paddle {

/**
 * The multibox loss layer for a SSD detection task.
 * The loss is composed by the location loss and the confidence loss.
 * The location loss is a smooth L1 loss and the confidence loss is
 * a softmax loss.
 * - Input: This layer needs four input layers: The first input layer
 *          is the priorbox layer and the second layer is a label layer.
 *          The rest two input layers are convolution layers for generating
 *          bbox location offset and the classification confidence.
 * - Output: The Single Shot Multibox Detection loss value.
 * Reference:
 *    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
 *    Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector
 */

class MultiBoxLossLayer : public CostLayer {
 public:
  explicit MultiBoxLossLayer(const LayerConfig& config) : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost) {}

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad) {}

 protected:
  inline LayerPtr getPriorBoxLayer() { return inputLayers_[0]; }
  inline LayerPtr getLabelLayer() { return inputLayers_[1]; }
  inline LayerPtr getLocInputLayer(size_t index) {
    return inputLayers_[2 + index];
  }
  inline LayerPtr getConfInputLayer(size_t index) {
    return inputLayers_[2 + inputNum_ + index];
  }

 protected:
  size_t numClasses_;
  real overlapThreshold_;
  real negPosRatio_;
  real negOverlap_;
  size_t inputNum_;
  size_t backgroundId_;

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
  MatrixPtr locGTData_;
  IVectorPtr confGTData_;

  MatrixPtr locBuffer_;
  MatrixPtr confBuffer_;
  MatrixPtr locDiff_;
  MatrixPtr confProb_;

  MatrixPtr labelCpuValue_;
  MatrixPtr priorCpuValue_;
  MatrixPtr locCpuBuffer_;
  MatrixPtr confCpuBuffer_;
  MatrixPtr locTmpBuffer_;
  MatrixPtr confTmpBuffer_;
};

}  // namespace paddle
