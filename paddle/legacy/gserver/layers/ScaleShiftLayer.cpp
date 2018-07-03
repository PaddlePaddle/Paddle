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

#include "Layer.h"

namespace paddle {

/**
 * A layer applies a linear transformation to each element in each row of
 * the input matrix. For each element, the layer first re-scale it and then
 * adds a bias to it.
 *
 * \f[
 *    y = wx + b
 * \f]
 *
 * Here, w is the scale and b is the bias. Both w and b are trainable scalars.
 *
 */

class ScaleShiftLayer : public Layer {
 protected:
  std::unique_ptr<Weight> scale_;
  std::unique_ptr<Weight> offset_;

 public:
  explicit ScaleShiftLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(scale_shift, ScaleShiftLayer);

bool ScaleShiftLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 1U);
  scale_.reset(new Weight(1, 1, parameters_[0]));
  if (biasParameter_.get() != NULL) {
    offset_ = std::unique_ptr<Weight>(new Weight(1, 1, biasParameter_));
  }
  return true;
}

void ScaleShiftLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);
  resetOutput(inV->getHeight(), inV->getWidth());
  MatrixPtr outV = getOutputValue();
  real scaleValue = scale_->getW()->getElement(0, 0);
  outV->mulScalar(*inV, scaleValue);
  if (offset_) {
    real offsetValue = offset_->getW()->getElement(0, 0);
    outV->add(offsetValue);
  }
}

void ScaleShiftLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV = getInputValue(0);
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr outV = getOutputValue();
  MatrixPtr outG = getOutputGrad();

  /* Calculate the parameter gradient for the current layer */
  if (scale_->getWGrad()) {
    MatrixPtr rowSumMtx;
    Matrix::resizeOrCreate(rowSumMtx, outG->getHeight(), 1, false, useGpu_);
    // this_i = scaleDest * this_i + scaleSum * \sum_j b_{ij} * c_{ij}
    rowSumMtx->sumOfProducts(
        /* b= */ *inV, /* c= */ *outG, /* scaleSum= */ 1, /* scaleDest= */ 0.);
    // this_i = scaleDest * this_i + scaleSum * \sum_j b_{ji}
    scale_->getWGrad()->sumCols(
        /* b= */ *rowSumMtx, /* scaleSum= */ 1., /* scaleDest= */ 1.);
    scale_->getParameterPtr()->incUpdate(callback);
  }
  if (offset_ && offset_->getWGrad()) {
    MatrixPtr rowSumMtx;
    Matrix::resizeOrCreate(rowSumMtx, outG->getHeight(), 1, false, useGpu_);
    rowSumMtx->sumRows(*outG, 1., 0.);
    offset_->getWGrad()->sumCols(*rowSumMtx, 1., 1.);
    offset_->getParameterPtr()->incUpdate(callback);
  }

  /* Calculate the input layers error */
  if (inG) {
    real scaleValue = scale_->getW()->getElement(0, 0);
    inG->add(*outG, scaleValue);
  }
}

}  // namespace paddle
