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
 * A layer for L2 normalization in each row,
 * \f[
 *   out[i] = \frac{in[i]}{\sqrt{\sum_{k=1}^N in[k]^{2}}}
 * \f]
 * where the size of \f$in\f$ is (batchSize x dataDim),
 * and the size of \f$out\f$ is (batchSize x dataDim).
 */

class RowL2NormLayer : public Layer {
 protected:
  MatrixPtr inSquare_;
  MatrixPtr l2NormReciprocal_;
  MatrixPtr dotSum_;

 public:
  explicit RowL2NormLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(row_l2_norm, RowL2NormLayer);

bool RowL2NormLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1U);

  return true;
}

void RowL2NormLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);

  /* malloc memory for the output_ if necessary */
  size_t batchSize = inV->getHeight();
  size_t dataDim = getSize();
  CHECK_EQ(dataDim, inV->getWidth());
  resetOutput(batchSize, dataDim);
  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(inSquare_, batchSize, dataDim, false, useGpu_);
  inV->square2(*inSquare_);
  Matrix::resizeOrCreate(l2NormReciprocal_, batchSize, 1, false, useGpu_);
  inSquare_->rowSum(*l2NormReciprocal_);
  l2NormReciprocal_->sqrt2(*l2NormReciprocal_);
  l2NormReciprocal_->scalarDiv(*l2NormReciprocal_, 1.0);
  outV->rowScale(0, *inV, *l2NormReciprocal_);
}

void RowL2NormLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV = getInputValue(0);
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr outV = getOutputValue();
  MatrixPtr outG = getOutputGrad();
  size_t batchSize = inV->getHeight();

  // inG[ij] += outG[ij] / l2NormReciprocal
  // inG[ij] += -inV[ij] * l2NormReciprocal * l2NormReciprocal * DotMul(outG[i],
  // inV[i])
  if (inG) {
    Matrix::resizeOrCreate(dotSum_, batchSize, 1, false, useGpu_);
    dotSum_->zeroMem();
    dotSum_->rowDotMul(0, *outG, *outV);
    dotSum_->dotMul(*dotSum_, *l2NormReciprocal_);
    dotSum_->dotMul(*dotSum_, *l2NormReciprocal_);
    inSquare_->rowScale(0, *inV, *dotSum_);
    inG->sub(*inSquare_);
    inG->addRowScale(0, *outG, *l2NormReciprocal_);
  }
}

}  // namespace paddle
