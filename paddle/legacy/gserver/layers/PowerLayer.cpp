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
 * This layer applys a power function to a vector element-wise,
 * which is used in NEURAL TURING MACHINE.
 * \f[
 *   y = x^w
 * \f]
 * where \f$x\f$ is a input vector, \f$w\f$ is scalar weight,
 * and output \f$y\f$ is a vector.
 *
 * The config file api is power_layer.
 */

class PowerLayer : public Layer {
 protected:
  MatrixPtr tmpMtx;

 public:
  explicit PowerLayer(const LayerConfig& config) : Layer(config) {}

  ~PowerLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(power, PowerLayer);

bool PowerLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  return true;
}

void PowerLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV1->getHeight();
  size_t dataDim = inV1->getWidth();

  CHECK_EQ(getSize(), dataDim);
  CHECK_EQ(1U, inV0->getWidth());
  CHECK_EQ(batchSize, inV0->getHeight());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();

  {
    REGISTER_TIMER_INFO("FwPowerTimer", getName().c_str());
    outV->rowPow(0, *inV1, *inV0);
  }
}

void PowerLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);
  MatrixPtr outV = getOutputValue();
  MatrixPtr outG = getOutputGrad();

  size_t batchSize = inV1->getHeight();
  size_t dataDim = inV1->getWidth();

  {
    REGISTER_TIMER_INFO("BwPowerTimer", getName().c_str());
    Matrix::resizeOrCreate(tmpMtx, batchSize, dataDim, false, useGpu_);

    if (inG0) {
      tmpMtx->log2(*inV1);
      tmpMtx->dotMul(*tmpMtx, *outV);

      // inG0 += outG .* (log(inV1) * outV)
      inG0->rowDotMul(0, *outG, *tmpMtx);
    }

    if (inG1) {
      // tmp = (outV / inV1) * inV0
      tmpMtx->dotDiv(*outV, *inV1);
      tmpMtx->rowScale(0, *tmpMtx, *inV0);

      inG1->addDotMul(*outG, *tmpMtx, 1, 1);
    }
  }
}

}  // namespace paddle
