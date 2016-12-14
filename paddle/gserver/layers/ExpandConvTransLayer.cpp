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

#include "ExpandConvTransLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

/* The implementation of the convTransLayer is basically a swap of forward and
 * backward of the original convLayer.
 * The variable naming follows the convention of the convLayer.
 * */

namespace paddle {

REGISTER_LAYER(exconvt, ExpandConvTransLayer);

bool ExpandConvTransLayer::init(const LayerMap &layerMap,
                                const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ExpandConvBaseLayer::init(layerMap, parameterMap);

  return true;
}

void ExpandConvTransLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getOutputSize());

  MatrixPtr output = nullptr;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    output = prevLayer->getOutputValue();
    REGISTER_TIMER_INFO("shrinkFwd", getName().c_str());
    bpropActs(output, getOutputValue(), i);
  }

  /* add the bias-vector */
  if (biases_.get()) {
    if (sharedBiases_) {
      addSharedBias();
    } else {
      addUnsharedBias();
    }
  }

  /* activation */
  forwardActivation();
}

void ExpandConvTransLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr imageGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    bpropBiases(imageGrad);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    /* First, calculate the input layers error */
    for (size_t off = 0; off < imageGrad->getHeight(); off++) {
      if (getPrev(i)->getOutputGrad()) {
        expandFwdOnce(imageGrad, getPrev(i)->getOutputGrad(), i, off);
      }
    }
    if (weights_[i]->getWGrad()) {
      /* Then, calculate the W-gradient for the current layer */
      bpropWeights(imageGrad, getPrev(i)->getOutputValue(), i);
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
