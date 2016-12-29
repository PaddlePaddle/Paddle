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
  return true;
}

std::vector<ParameterPtr>& CaffeLayer::std::vector<ParameterPtr>&
initParamHook() {
  // set bottom
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto blob = new Blob<float>();
    blob->Reshape(layerConfig2BlobShape(1, getPrevConfig(i)));
    bot_.push_back(blob);
  }

  // set top
  top_.push_back(new Blob<float>());

  // the parameter shape can only be obtained after setup
  caffeOp_->SetUp(bot_, top_);

  for (int i = 0; i < caffeOp_->blobs().size(); ++i) {
    auto& w = caffeOp_->blobs()[i];
    int h = w->shape()[0];
    int w = w->count(1);
    ParameterConfig* conf = new ParameterConfig();
    std::string pname = getName() + "_.w" + std::to_string(i);
    conf->set_name(pname);
    conf->set_size(w->count());
    conf->add_dims(h);
    conf->add_dims(w);
    paramconfig_.emplace_back(conf);
    auto parameter = std::make_shared<Parameter>(paramconfig_[0].get(),
                                                 useGpu_,
                                                 /*initialize=*/false);
    parameters_.push_back(parameter);
    Weight* w = new Weight(h, w, parameter);
    weights_.emplace_back(w);
  }
  return parameters_;
}

void CaffeLayer::caffeLayerSetup(int curBatchSize) {
  // Since some arguments used in Forward and Backward are initialized
  // in SetUp in Caffe. The batch size may change in each mini-batch in
  // Paddle. So, these arguments may change, it needs to reset them
  // by SetUp functions.
  // Whether it need to reset depending on the changes of batch size now.
  // If the frameHeight and frameWidth changes, it also needs to reset.
  if (curBatchSize != batchSize_) {
    caffeOp_->SetUp(bot_, top_);
    batchSize_ = curBatchSize;
    // set output
    setDataToArg(top_[0], getOutput(), useGpu_);
  }
}

void CaffeLayer::forward(PassType passType) {
  Layer::forward(passType);
  setMode(useGpu_);

  int batchSize = getInput(0).getBatchSize();

  // set bottom data
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    setDataToBlob(getInput(i), bot_[i], useGpu_);
  }
  caffeLayerSetup(batchSize);

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
