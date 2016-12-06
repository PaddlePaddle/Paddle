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

#include "CaffeLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(caffe, CaffeLayer);

bool CaffeLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // create caffe layer
  auto param = getLayerParameter(config_.caffe_conf().prototxt());
  caffeOp_ = LayerRegistry<real>::CreateLayer(*param);

  propagateDown_.resize(inputLayers_.size());

  // set for test
  bot_.resize(inputLayers_.size());
  top_.resize(1);

  return true;
}

void CaffeLayer::caffeLayerSetup() {
  if (!setup_) {
    setup_ = true;
    caffeOp_->SetUp(bot_, top_);
  }
}

void CaffeLayer::forward(PassType passType) {
  Layer::forward(passType);
  setMode(useGpu_);

  // set data of bot_, top_
  // Argument2CaffeBlob

  caffeLayerSetup();

  // set data of weight
  // share memory between caffe blobs and paddle parameter.

  caffeOp_->Forward(bot_, top_);

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void CaffeLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  // set diff of bot_, top_
  // Argument2CaffeBlob
  // set propagateDown_

  caffeOp_->Backward(top_, propagateDown_, bot_);

  // might need to add bot_ diff to input grad.

  // parameter updater
}

}  // namespace paddle
