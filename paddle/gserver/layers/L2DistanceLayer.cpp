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

#include "L2DistanceLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(l2_distance, L2DistanceLayer);

bool L2DistanceLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2UL) << "The L2DistanceLayer accepts two and "
                                     << "only two inputs.";
  CHECK_EQ(getSize(), 1UL) << "The output dimensionality of L2DistanceLayer "
                           << "is fixed to be 1.";

  return true;
}

void L2DistanceLayer::forward(PassType passType) {
  Layer::forward(passType);

  const auto inV1 = getInputValue(0);
  const auto inV2 = getInputValue(1);

  CHECK(inV1 && inV2);
  CHECK_EQ(inV1->getHeight(), inV2->getHeight())
      << "The height of two inputs of this layer must be the same.";
  CHECK_EQ(inV1->getWidth(), inV2->getWidth())
      << "The width of two inputs of this layer must be the same.";

  int batchSize = inV1->getHeight();
  int output_dim = getSize();
  {
    REGISTER_TIMER_INFO("L2DistanceBpAtvTimer", getName().c_str());
    reserveOutput(batchSize, output_dim);
    auto outV = getOutputValue();
    CHECK(outV) << "The output matrix should not be null.";

    Matrix::resizeOrCreate(
        inputSub_, inV1->getHeight(), inV1->getWidth(), false, useGpu_);

    inputSub_->assign(*inV1);
    inputSub_->sub(*inV2);
    outV->sumOfProducts(*inputSub_, *inputSub_, 1, 0);
    outV->sqrt2(*outV);
  }
}

void L2DistanceLayer::backward(const UpdateCallback& callback) {
  const auto outG = getOutputGrad();
  const auto outV = getOutputValue();
  CHECK(outG && outV);

  auto inGrad1 = getInputGrad(0);
  auto inGrad2 = getInputGrad(1);

  {
    REGISTER_TIMER_INFO("L2DistanceBpAtvTimer", getName().c_str());

    if (inGrad1 || inGrad2) {
      outV->scalarDiv(*outV, 1.);
      outV->dotMul(*outG, *outV);
    }

    if (inGrad1) inGrad1->addRowScale(0, *inputSub_, *outV);

    if (inGrad2) {
      inputSub_->mulScalar(-1.);
      inGrad2->addRowScale(0, *inputSub_, *outV);
    }
  }
}

}  // namespace paddle
