/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
 * A layer for each element in a channel, multiplying with an element of a
 * vector.
 *
 * \f[
 *   y[i][c * size + j] = w[i][c] * x[i][c * size + j]
 * \f]
 *
 * where c is [0, number of channel). The size is the number of elements in a
 * channel, taking an image as input for example, the size is height * width.
 * The range of j is [0, size).
 *
 * The config file api is broadcast_scale_layer.
 */

class BroadcastScaleLayer : public Layer {
public:
  explicit BroadcastScaleLayer(const LayerConfig& config) : Layer(config) {}

  ~BroadcastScaleLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(broadcast_scale_layer, BroadcastScaleLayer);

bool BroadcastScaleLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  return true;
}

void BroadcastScaleLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);
  MatrixPtr weightV = getInputValue(1);

  size_t batchSize = inV->getHeight();
  size_t dataDim = inV->getWidth();
  size_t channelNum = weightV->getWidth();

  CHECK_EQ(dataDim, getSize());
  CHECK_EQ(weightV->getHeight(), batchSize);
  CHECK_EQ(dataDim % channelNum, 0UL);

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwBroadcastScaleTimer", getName().c_str());
    outV->addBroadcastMul(*inV, *weightV);
  }
}

void BroadcastScaleLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV = getInputValue(0);
  MatrixPtr weightV = getInputValue(1);
  MatrixPtr inGrad = getInputGrad(0);
  MatrixPtr weightGrad = getInputGrad(1);
  MatrixPtr outGrad = getOutputGrad();

  {
    REGISTER_TIMER_INFO("BwBroadcastScaleTimer", getName().c_str());

    if (inGrad) {
      inGrad->addBroadcastMul(*outGrad, *weightV);
    }

    if (weightGrad) {
      weightGrad->rowDotMul(0, *outGrad, *inV);
    }
  }
}

}  // namespace paddle
