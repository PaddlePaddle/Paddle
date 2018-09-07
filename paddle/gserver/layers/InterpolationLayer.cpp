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
 * A layer for linear interpolation with two inputs,
 * which is used in NEURAL TURING MACHINE.
 * \f[
 *   y.row[i] = w[i] * x_1.row[i] + (1 - w[i]) * x_2.row[i]
 * \f]
 * where \f$x_1\f$ and \f$x_2\f$ are two (batchSize x dataDim) inputs,
 * \f$w\f$ is (batchSize x 1) weight vector,
 * and \f$y\f$ is (batchSize x dataDim) output.
 *
 * The config file api is interpolation_layer.
 */

class InterpolationLayer : public Layer {
 protected:
  /// weightLast = 1 - weight
  MatrixPtr weightLast_;
  MatrixPtr tmpMatrix;

 public:
  explicit InterpolationLayer(const LayerConfig& config) : Layer(config) {}

  ~InterpolationLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(interpolation, InterpolationLayer);

bool InterpolationLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(3U, inputLayers_.size());

  return true;
}

void InterpolationLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr weightV = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inV2 = getInputValue(2);

  size_t batchSize = inV1->getHeight();
  size_t dataDim = inV1->getWidth();

  CHECK_EQ(dataDim, getSize());
  CHECK_EQ(dataDim, inV2->getWidth());
  CHECK_EQ(batchSize, inV1->getHeight());
  CHECK_EQ(batchSize, inV2->getHeight());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(weightLast_, batchSize, 1, false, useGpu_);
  weightLast_->one();
  weightLast_->sub(*weightV);

  REGISTER_TIMER_INFO("FwInterpTimer", getName().c_str());
  // outV = inV1 * weight + inV2 * weightLast
  outV->addRowScale(0, *inV1, *weightV);
  outV->addRowScale(0, *inV2, *weightLast_);
}

void InterpolationLayer::backward(const UpdateCallback& callback) {
  MatrixPtr outG = getOutputGrad();
  MatrixPtr weightV = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inV2 = getInputValue(2);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);
  MatrixPtr inG2 = getInputGrad(2);

  size_t batchSize = inV1->getHeight();
  size_t dataDim = inV1->getWidth();

  REGISTER_TIMER_INFO("BwInterpTimer", getName().c_str());

  if (inG0) {
    Matrix::resizeOrCreate(tmpMatrix, batchSize, dataDim, false, useGpu_);

    // inG0 += outG .* (inV1 - inV2)
    tmpMatrix->sub(*inV1, *inV2);
    inG0->rowDotMul(0, *outG, *tmpMatrix);
  }

  if (inG1) {
    // inG1 += outG * weight
    inG1->addRowScale(0, *outG, *weightV);
  }

  if (inG2) {
    // inG2 += outG * weightLast
    inG2->addRowScale(0, *outG, *weightLast_);
  }
}

}  // namespace paddle
