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
 * @brief A layer for applying a slope and an intercept to the input
 * element-wise.
 * This layer is used in NEURAL TURING MACHINE.
 * @note There is no activation and weight in this layer.
 *
 * \f[
 *    y = ax + b
 * \f]
 *
 * Here, a is scale and b is offset, which are provided as attributes of the
 * layer.
 *
 * The config file api is slope_intercept_layer.
 */

class SlopeInterceptLayer : public Layer {
 public:
  explicit SlopeInterceptLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(slope_intercept, SlopeInterceptLayer);

bool SlopeInterceptLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1U);

  return true;
}

void SlopeInterceptLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);

  /* malloc memory for the output_ if necessary */
  size_t batchSize = inV->getHeight();
  size_t size = getSize();

  CHECK_EQ(size, inV->getWidth());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwSlopeInterceptTimer", getName().c_str());
    outV->mulScalar(*inV, config_.slope());
    outV->add(config_.intercept());
  }
}

void SlopeInterceptLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr outG = getOutputGrad();

  if (inG) {
    REGISTER_TIMER_INFO("BwSlopeInterceptTimer", getName().c_str());
    inG->add(*outG, config_.slope());
  }
}

}  // namespace paddle
