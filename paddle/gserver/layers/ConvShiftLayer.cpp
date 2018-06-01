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
 * @brief A layer for circular convluation of two vectors,
 * which is used in NEURAL TURING MACHINE.
 * - Input: two vectors, the first is data (batchSize x dataDim)
 * the second is shift weights (batchSize x shiftDim)
 * - Output: a vector (batchSize x dataDim)
 * Assumed that:
 * - a[in]: contains M elements.
 * - b[in]: contains N elements (N should be odd).
 * - c[out]: contains M elements.
 *
 * \f[
 *     c[i] = \sum_{j=-(N-1)/2}^{(N-1)/2}a_{i+j} * b_{j}
 * \f]
 *
 * In this formula:
 *  - a's index is computed modulo M.
 *  - b's index is comupted modulo N.
 *
 * The config file api is conv_shift_layer.
 */

class ConvShiftLayer : public Layer {
 public:
  explicit ConvShiftLayer(const LayerConfig& config) : Layer(config) {}

  ~ConvShiftLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(conv_shift, ConvShiftLayer);

bool ConvShiftLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  return true;
}

void ConvShiftLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV0->getHeight();
  size_t dataDim = inV0->getWidth();

  CHECK_EQ(batchSize, inV1->getHeight());
  CHECK_EQ(dataDim, getSize());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();

  REGISTER_TIMER_INFO("FwConvShiftTimer", getName().c_str());
  outV->circularConv(*inV0, *inV1);
}

void ConvShiftLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr outG = getOutputGrad();
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);

  REGISTER_TIMER_INFO("BwConvShiftTimer", getName().c_str());

  if (inG0 && inG1) {
    outG->circularConvDerivative(*outG, *inV0, *inV1, *inG0, *inG1);
  } else {
    CHECK(!inG0 || !inG1) << "Not supported";
  }
}

}  // namespace paddle
