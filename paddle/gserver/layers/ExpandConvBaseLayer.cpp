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

#include "ExpandConvBaseLayer.h"

#include "paddle/utils/Logging.h"
namespace paddle {

bool ExpandConvBaseLayer::init(const LayerMap &layerMap,
                               const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
  }

  getOutputSize();

  return true;
}

size_t ExpandConvBaseLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  return layerSize;
}

void ExpandConvBaseLayer::addSharedBias() {
  size_t mapW = getOutputSize() / numFilters_;
  size_t mapH = getOutputValue()->getElementCnt() / mapW;
  MatrixPtr out =
      Matrix::create(getOutputValue()->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  out->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / numFilters_,
                          numFilters_);

  MatrixPtr bias = Matrix::create(biases_->getW()->getData(),
                                  1,
                                  biases_->getW()->getElementCnt(),
                                  false,
                                  useGpu_);
  transOutValue_->addBias(*bias, 1.0f);

  transOutValue_->reshape(mapW, mapH);
  transOutValue_->transpose(out, false);  // false means no memory allocation

  out->clear();
  bias->clear();
}

void ExpandConvBaseLayer::addUnsharedBias() {
  MatrixPtr outValue = getOutputValue();
  MatrixPtr bias = Matrix::create(biases_->getW()->getData(),
                                  1,
                                  biases_->getW()->getElementCnt(),
                                  false,
                                  useGpu_);
  outValue->addBias(*bias, 1.0f);
}

void ExpandConvBaseLayer::bpropSharedBias(MatrixPtr biases, MatrixPtr v) {
  size_t mapW = getOutputSize() / numFilters_;
  size_t mapH = v->getElementCnt() / mapW;
  MatrixPtr vTmp = Matrix::create(v->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  vTmp->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / numFilters_,
                          numFilters_);
  biases->collectBias(*transOutValue_, 1.0f);
}

void ExpandConvBaseLayer::bpropBiases(MatrixPtr v) {
  MatrixPtr biases = Matrix::create(biases_->getWGrad()->getData(),
                                    1,
                                    biases_->getWGrad()->getElementCnt(),
                                    false,
                                    useGpu_);
  if (sharedBiases_) {
    bpropSharedBias(biases, v);
  } else {
    biases->collectBias(*v, 1.0f);
  }
  biases->clear();
}

}  // namespace paddle
