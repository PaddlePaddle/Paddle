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
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for sum-to-one normalization,
 * which is used in NEURAL TURING MACHINE.
 * \f[
 *   out[i] = \frac {in[i]} {\sum_{k=1}^N in[k]}
 * \f]
 * where \f$in\f$ is a (batchSize x dataDim) input vector,
 * and \f$out\f$ is a (batchSize x dataDim) output vector.
 *
 * The config file api is sum_to_one_norm_layer.
 */

class SumToOneNormLayer : public Layer {
 protected:
  /// reciprocalRowSum_ = \f$1 / \sum_{k=1}^N in[k]\f$
  MatrixPtr reciprocalRowSum_;
  /// dotSum = output_.grad \f$.*\f$ output_.value
  MatrixPtr dotSum_;

 public:
  explicit SumToOneNormLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(sum_to_one_norm, SumToOneNormLayer);

bool SumToOneNormLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1U);

  return true;
}

void SumToOneNormLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);

  /* malloc memory for the output_ if necessary */
  size_t batchSize = inV->getHeight();
  size_t dataDim = getSize();

  CHECK_EQ(dataDim, inV->getWidth());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwSumToOneNormTimer", getName().c_str());

    Matrix::resizeOrCreate(reciprocalRowSum_, batchSize, 1, false, useGpu_);
    inV->rowSum(*reciprocalRowSum_);

    // todo: matrix checks
    CHECK_GT(reciprocalRowSum_->getMin(), 0.0);

    reciprocalRowSum_->scalarDiv(*reciprocalRowSum_, 1.0);

    // outV = inV * reciprocalRowSum
    outV->rowScale(0, *inV, *reciprocalRowSum_);
  }
}

void SumToOneNormLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV = getInputValue(0);
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr outV = getOutputValue();
  MatrixPtr outG = getOutputGrad();

  size_t batchSize = inV->getHeight();

  if (inG) {
    REGISTER_TIMER_INFO("BwSumToOneTimer", getName().c_str());

    Matrix::resizeOrCreate(dotSum_, batchSize, 1, false, useGpu_);

    // dotSum = outG .* outV
    dotSum_->zeroMem();
    dotSum_->rowDotMul(0, *outG, *outV);

    // inG += -1 * (dotSum / rowSum)
    dotSum_->dotMul(*dotSum_, *reciprocalRowSum_);
    inG->rowAdd(0, *inG, *dotSum_, -1.0);
    // inG += outG * (1/rowSum)
    inG->addRowScale(0, *outG, *reciprocalRowSum_);
  }
}

}  // namespace paddle
