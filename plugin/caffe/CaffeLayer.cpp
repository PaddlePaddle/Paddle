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

#include "CaffeLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {
using ::caffe::Blob;

REGISTER_LAYER(caffe, CaffeLayer);

bool CaffeLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  // create caffe layer
  auto param = getLayerParameter(config_.caffe_conf().prototxt());
  caffeOp_ = LayerRegistry<real>::CreateLayer(*param);
  weightNums_ = config_.caffe_conf().num_weights();
  inputNums_ = config_.caffe_conf().num_input();
  CHECK_EQ(weightNums_, static_cast<int>(parameters_.size()));
  resetParameters();
  propagateDown_.assign(inputLayers_.size(), true);
  top_.push_back(new Blob<real>());

  return true;
}

void CaffeLayer::resetParameters() {
  // set bottom
  for (int i = 0; i != inputNums_; ++i) {
    auto blob = new Blob<real>();
    blob->Reshape(layerConfigToBlobShape(1, getPrevConfig(i)));
    bot_.push_back(blob);
  }

  // set top
  std::vector<Blob<real>*> top;
  top.push_back(new Blob<real>());

  // the parameter shape can only be obtained after setup
  caffeOp_->SetUp(bot_, top);
  CHECK_LE(top.size(), 1);

  CHECK_EQ(weightNums_, static_cast<int>(caffeOp_->blobs().size()));
  for (size_t i = 0; i < caffeOp_->blobs().size(); ++i) {
    auto& wblob = caffeOp_->blobs()[i];
    std::vector<int> dim(2);
    dim[0] = wblob->shape()[0];
    dim[1] = wblob->count(1);
    ParameterConfig* conf = &(parameters_[i]->getConfig());
    conf->set_initial_strategy(PARAMETER_INIT_SKIP);
    parameters_[i]->resize(wblob->count(), dim);
    copyBlobToParameter(
        VALUE, caffeOp_->blobs()[i].get(), parameters_[i], useGpu_);
    wei_.push_back(new ::caffe::Blob<real>());
    parameterToBlob(VALUE, parameters_[i], wei_[i], wblob->shape(), useGpu_);
  }

  CHECK_EQ(caffeOp_->blobs().size(), wei_.size());
  for (size_t i = 0; i < wei_.size(); ++i) {
    caffeOp_->blobs()[i].reset(wei_[i]);
    ParameterConfig conf = parameters_[i]->getConfig();
    bool prop = (!conf.is_static()) && conf.learning_rate() > 0.0;
    caffeOp_->set_param_propagate_down(i, prop);
  }
  for (auto blob : top) delete blob;
}

void CaffeLayer::caffeLayerSetup(int curBatchSize) {
  // Since some arguments used in Forward and Backward are initialized
  // in SetUp in Caffe. The batch size may change in each mini-batch in
  // Paddle. So, these arguments may change, it needs to reset them
  // by SetUp functions.
  // Whether it need to reset depending on the changes of batch size now.
  // If the frameHeight and frameWidth changes, it also needs to reset.
  if (curBatchSize != batchSize_) {
    std::vector<Blob<real>*> top;
    top.push_back(new Blob<real>());
    caffeOp_->SetUp(bot_, top);
    batchSize_ = curBatchSize;
    int width = top[0]->count(1);
    resetOutput(curBatchSize, width);
    argToBlob(VALUE, getOutput(), top_[0], useGpu_);
    for (auto blob : top) delete blob;
  }
}

void CaffeLayer::forward(PassType passType) {
  Layer::forward(passType);
  setMode(useGpu_);

  int batchSize = getInput(0).getBatchSize();

  for (int i = 0; i != inputNums_; ++i) {
    // setting bottom. The memory is shared between Input(i) and bot_[i].
    argToBlob(VALUE, getInput(i), bot_[i], useGpu_);
  }
  caffeLayerSetup(batchSize);

  caffeOp_->Forward(bot_, top_);

  // Output shared memory with top_[0]
  blobToArg(VALUE, top_[0], getOutput(), useGpu_);

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

  // set top diff
  argToBlob(GRAD, getOutput(), top_[0], useGpu_);

  // set bottom diff
  for (int i = 0; i != inputNums_; ++i) {
    argToBlob(GRAD, getInput(i), bot_[i], useGpu_);
    if (!getInputGrad(i)) {
      propagateDown_[i] = false;
    }
  }

  // set blob diff
  for (size_t i = 0; i < wei_.size(); ++i) {
    if (parameters_[i]->getBuf(PARAMETER_GRADIENT)) {
      parameterToBlob(GRAD, parameters_[i], wei_[i], wei_[i]->shape(), useGpu_);
    }
  }

  caffeOp_->Backward(top_, propagateDown_, bot_);

  // might need to add bot_ diff to input grad.
  for (int i = 0; i < weightNums_; ++i) {
    parameters_[i]->incUpdate(callback);
  }
}

}  // namespace paddle
