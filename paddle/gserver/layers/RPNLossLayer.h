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
 * The loss layer for region proposal in Faster R-CNN.
 * The loss is composed by the location loss and the confidence loss.
 * The location loss is a smooth L1 loss and the confidence loss is
 * a softmax loss.
 * - Input: This layer needs four input layers: The first input layer
 *          is the anchor-box layer and the second layer is a label layer.
 *          The rest two input layers are convolution layers for generating
 *          bbox location offset and the classification confidence.
 * - Output: The Region Proposal Networks loss value.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 */

class RPNLossLayer : public CostLayer {
public:
  explicit RPNLossLayer(const LayerConfig& config)
      : CostLayer(config), rand_(0, 1) {}

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

  void bboxOverlaps(const std::vector<std::vector<real>>& anchorBoxes,
                    const std::vector<std::vector<real>>& gtBBoxes,
                    std::vector<real>& overlaps);

  std::pair<size_t, size_t> labelAnchors(
      const std::vector<std::vector<real>>& anchorBoxes,
      const std::vector<std::vector<real>>& gtBBoxes,
      const std::vector<real>& overlaps,
      const real posOverlapThreshold,
      const real negOverlapThreshold,
      std::vector<int>& matchIndices,
      std::vector<int>& labels);

  template <typename T>
  void sampleAnchors(
      std::vector<T>& allLabels, T label, T disabledLable, size_t m, size_t n);

  pair<size_t, size_t> generateMatchIndices(
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
      std::vector<std::vector<int>>* negIndicesVecPtr);

  void encodeTarget(const std::vector<real>& anchorBox,
                    const std::vector<real>& gtBBox,
                    std::vector<real>& target);

protected:
  real posOverlapThreshold_;
  real negOverlapThreshold_;
  size_t rpnBatchSize_;
  real rpnFgRatio_;
  real lossRatio_;
  size_t numClasses_;
  size_t inputNum_;
  size_t backgroundId_;
  std::uniform_real_distribution<real> rand_;

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
