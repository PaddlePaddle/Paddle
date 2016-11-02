/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "ExpandConvLayer.h"

namespace paddle {

REGISTER_LAYER(exconv, ExpandConvLayer);

bool ExpandConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ExpandConvBaseLayer::init(layerMap, parameterMap);
  return true;
}

size_t ExpandConvLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  subN_.clear();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    subN_.push_back(outputH_[i] * outputW_[i]);
  }
  return layerSize;
}

void ExpandConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one colum, and the
   *   transOutValue correspond sample to one row */
  int batchSize = inputLayers_[0]->getOutputValue()->getWidth();
  batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getOutputSize());

  MatrixPtr image = nullptr;
  MatrixPtr outV = getOutputValue();
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    image = prevLayer->getOutputValue();
    for (size_t off = 0; off < image->getHeight(); off++) {
      REGISTER_TIMER_INFO("expandFwdOnce", getName().c_str());
      expandFwdOnce(image, outV, i, off);
    }
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


void ExpandConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr outGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    bpropBiases(outGrad);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    /* First, calculate the input layers error */
    if (getPrev(i)->getOutputGrad()) {
      bpropActs(outGrad, getPrev(i)->getOutputGrad(), i);
    }
    if (weights_[i]->getWGrad()) {
      /* Then, calculate the W-gradient for the current layer */
      bpropWeights(getPrev(i)->getOutputValue(), outGrad, i);
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
