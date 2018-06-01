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
 * A layer for each row of a matrix, multiplying with a element of a vector,
 * which is used in NEURAL TURING MACHINE.
 * \f[
 *   y.row[i] = w[i] * x.row[i]
 * \f]
 * where \f$x\f$ is (batchSize x dataDim) input, \f$w\f$ is
 * (batchSize x 1) weight vector, and \f$y\f$ is (batchSize x dataDim) output.
 *
 * The config file api is scaling_layer.
 */

class ScalingLayer : public Layer {
 public:
  explicit ScalingLayer(const LayerConfig& config) : Layer(config) {}

  ~ScalingLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(scaling, ScalingLayer);

bool ScalingLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  return true;
}

void ScalingLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr weightV = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV1->getHeight();
  size_t dataDim = inV1->getWidth();

  CHECK_EQ(dataDim, getSize());
  CHECK_EQ(weightV->getWidth(), 1U);
  CHECK_EQ(weightV->getHeight(), batchSize);

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwScalingTimer", getName().c_str());
    // outV += inV1 * weight
    outV->addRowScale(0, *inV1, *weightV);
  }
}

void ScalingLayer::backward(const UpdateCallback& callback) {
  MatrixPtr weightV = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);
  MatrixPtr outG = getOutputGrad();

  {
    REGISTER_TIMER_INFO("BwScalingTimer", getName().c_str());

    if (inG0) {
      // inG0 += outG .* inV1
      inG0->rowDotMul(0, *outG, *inV1);
    }

    if (inG1) {
      // inG1 += outG * weight;
      inG1->addRowScale(0, *outG, *weightV);
    }
  }
}

}  // namespace paddle
