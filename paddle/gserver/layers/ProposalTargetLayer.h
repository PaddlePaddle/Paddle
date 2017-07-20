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

#include "Layer.h"

namespace paddle {

/**
 * A layer used by Fast(er) R-CNN to generate ROIs' data for training.
 * - Input: This layer needs four input layers: The first input layer
 *          is the ProposalLayer and the second layer is a label layer.
 * - Output: The ROIs' data, including locations, labels and targets.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 */

class ProposalTargetLayer : public Layer {
public:
  explicit ProposalTargetLayer(const LayerConfig& config)
      : Layer(config), rand_(0, 1) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override {}

protected:
  void bboxOverlaps(const std::vector<std::vector<real>>& anchorBoxes,
                    const std::vector<std::vector<real>>& gtBBoxes,
                    std::vector<real>& overlaps);

  std::pair<size_t, size_t> labelBBoxes(
      const std::vector<std::vector<real>>& anchorBoxes,
      const std::vector<std::vector<real>>& gtBBoxes,
      const std::vector<real>& overlaps,
      const real posOverlapThreshold,
      const real negOverlapThreshold,
      std::vector<int>& matchIndices,
      std::vector<int>& labels);

  template <typename T>
  void sampleBBoxes(
      std::vector<T>& allLabels, T label, T disabledLable, size_t m, size_t n);

  std::pair<size_t, size_t> generateMatchIndices(
      const Matrix& priorValue,
      const Matrix& gtValue,
      const int* gtStartPosPtr,
      const size_t gtBBoxDim,
      const size_t seqNum,
      const real posOverlapThreshold,
      const real negOverlapThreshold,
      const size_t boxBatchSize,
      const real boxFgRatio,
      std::vector<std::vector<int>>* priorBBoxIdxsVecPtr,
      std::vector<std::vector<int>>* matchIndicesVecPtr);

  void encodeTarget(const std::vector<real>& priorBBox,
                    const std::vector<real>& gtBBox,
                    std::vector<real>& target);

protected:
  real posOverlapThreshold_;
  real negOverlapThreshold_;
  size_t boxBatchSize_;
  real boxFgRatio_;
  size_t numClasses_;
  size_t backgroundId_;
  std::uniform_real_distribution<real> rand_;

  std::vector<std::vector<int>> allPriorBBoxIdxs_;
  std::vector<std::vector<int>> allMatchIndices_;
};

}  // namespace paddle
