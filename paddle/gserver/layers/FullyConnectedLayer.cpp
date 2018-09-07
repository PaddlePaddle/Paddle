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

#include "FullyConnectedLayer.h"
#include <algorithm>
#include <vector>
#include "paddle/math/SparseMatrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(fc, FullyConnectedLayer);

bool FullyConnectedLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    size_t height = inputLayers_[i]->getSize();
    size_t width = getSize();

    // create a new weight
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), width * height);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), width * height);
    }
    Weight* w = new Weight(height, width, parameters_[i]);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  return true;
}

void FullyConnectedLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto* sparseParam =
        dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
    if (sparseParam) {
      MatrixPtr input = getInputValue(i);
      sparseParam->addRows(input);
    }
  }
}

void FullyConnectedLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto input = getInput(i);
    CHECK(input.value) << "The input of 'fc' layer must be matrix";
    REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
    i == 0 ? outV->mul(*input.value, *weights_[i]->getW(), 1, 0)
           : outV->mul(*input.value, *weights_[i]->getW(), 1, 1);
  }

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void FullyConnectedLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  bool syncFlag = hl_get_sync_flag();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the W-gradient for the current layer */
    if (weights_[i]->getWGrad()) {
      MatrixPtr input_T = getInputValue(i)->getTranspose();
      MatrixPtr oGrad = getOutputGrad();
      {
        REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
        weights_[i]->getWGrad()->mul(*input_T, *oGrad, 1, 1);
      }
    }

    // If callback does not change value, backprop error asynchronously so that
    // we can do the callback concurrently.
    hl_set_sync_flag(false);

    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      MatrixPtr weights_T = weights_[i]->getW()->getTranspose();
      REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      preGrad->mul(*getOutputGrad(), *weights_T, 1, 1);
    }

    hl_set_sync_flag(syncFlag);
    {
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
