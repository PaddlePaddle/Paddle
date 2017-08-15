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

#include "CostLayer.h"
#include "Layer.h"

namespace paddle {

/**
 * The loss layer in Fast(er) R-CNN. The loss is composed by the location
 * loss and the confidence loss. The location loss is a smooth L1 loss and
 * the confidence loss is a softmax loss.
 * - Input: This layer needs three input layers: The first input layer
 *          contains the ROIs' data and the rest two input layers are
 *          layers generating bbox location offset and the classification
 *          confidence.
 * - Output: The detection loss value.
 * Reference:
 *    Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
 *    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
 */

class RCNNLossLayer : public CostLayer {
public:
  explicit RCNNLossLayer(const LayerConfig& config) : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost) {}

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad) {}

protected:
  real lossRatio_;
  size_t numClasses_;
  size_t backgroundId_;

  std::vector<std::pair<size_t, size_t>> posROIs_;
  MatrixPtr locPosGT_;
  MatrixPtr locPosPred_;
  IVectorPtr confGT_;
  MatrixPtr confPred_;
};

}  // namespace paddle
