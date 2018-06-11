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
 * @brief A layer for computing the dot product of two vectors.
 * Input1: vector (batchSize * dim)
 * Input2: vector (batchSize * dim)
 * Output: a matrix: (batchSize * 1)
 */

class DotProdLayer : public Layer {
 public:
  explicit DotProdLayer(const LayerConfig& config) : Layer(config) {}

  ~DotProdLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(dot_prod, DotProdLayer);

bool DotProdLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);
  CHECK_EQ(1UL, getSize())
      << "The output dimensionality of this layer should be fixed to 1.";

  return true;
}

void DotProdLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV0->getHeight();
  CHECK_EQ(inV1->getHeight(), batchSize);
  CHECK_EQ(inV0->getWidth(), inV1->getWidth());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, 1);
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwDotProdTimer", getName().c_str());
    outV->sumOfProducts(*inV0, *inV1, 1, 0);
  }
}

void DotProdLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr outG = getOutputGrad();
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);

  {
    REGISTER_TIMER_INFO("BwDotProdTimer", getName().c_str());

    if (inG0) {
      inG0->addRowScale(0, *inV1, *outG);
    }

    if (inG1) {
      inG1->addRowScale(0, *inV0, *outG);
    }
  }
}

}  // namespace paddle
