/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MkldnnFcLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  if (!MkldnnLayer::init(layerMap, parameterMap)) {
    return false;
  }

  CHECK_EQ(inputLayers_.size(), 1) << "Only support one input layer yet!";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK(!parameters_[0]->isSparse()) << "Do not support sparse yet";

  // output size, cat not be changed
  oc_ = getSize();
  oh_ = 1;
  ow_ = 1;

  // input size can not change in FC
  iLayerSize_ = inputLayers_[0]->getSize();
  CHECK_EQ(parameters_[0]->getSize(), iLayerSize_ * oc_);

  // create weight
  weight_ =
      std::unique_ptr<Weight>(new Weight(oc_, iLayerSize_, parameters_[0], 0));
  initWgt();

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

void MkldnnFcLayer::initWgt() {
  // The weight_ is transposed from initial paddle weight
  MatrixPtr paddleWgt = Matrix::create(
      weight_->getW()->getData(), iLayerSize_, oc_, false, false);

  std::ostringstream ostr;
  paddleWgt->print(ostr);
  VLOG(DNN_BASE) << ostr.str();

  // Firstly in mkldnn, the matrix is transposed from initial paddle weight
  MatrixPtr paddleWgtT;
  paddleWgt->transpose(paddleWgtT, true);

  weight_->getW()->copyFrom(*paddleWgtT);
}

void MkldnnFcLayer::reshape() {
  const Argument& input = getInput(0);
  int batchSize = input.getBatchSize();
  if (bs_ == batchSize) {
    return;
  }
  bs_ = batchSize;
  ih_ = input.getFrameHeight();
  iw_ = input.getFrameWidth();
  if (ih_ == 0) {
    ih_ = 1;
  }
  if (iw_ == 0) {
    iw_ = 1;
  }
  CHECK_EQ(iLayerSize_, inputLayers_[0]->getSize());
  ic_ = iLayerSize_ / (ih_ * iw_);
  CHECK_EQ(size_t(ic_ * ih_ * iw_), iLayerSize_) << "not divisible";
  CHECK_EQ(size_t(oc_), getSize());

  // reset output
  output_.setFrameHeight(oh_);
  output_.setFrameWidth(ow_);
  resetOutput(bs_, oc_);
}

void MkldnnFcLayer::forward(PassType passType) {
  Layer::forward(passType);
  reshape();

  {
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
    real* input = getInputValue(0)->getData();
    real* output = getOutputValue()->getData();
    real* wgt = weight_->getW()->getData();
    bool hasBias = biases_ && biases_->getW();
    real* bias = hasBias ? biases_->getW()->getData() : NULL;
    mkldnnForwardFC(bs_, ic_, ih_, iw_, input, oc_, output, wgt, bias);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwActTimer", getName().c_str());
    forwardActivation();
  }
}

void MkldnnFcLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpActTimer", getName().c_str());
    backwardActivation();
  }

  bool hasBias = biases_ && biases_->getWGrad();
  {
    REGISTER_TIMER_INFO("mkldnn_bwdTimer", getName().c_str());
    real* inVal = getInputValue(0)->getData();
    real* inGrad =
        getInputGrad(0) != nullptr ? getInputGrad(0)->getData() : NULL;
    real* outGrad = getOutputGrad()->getData();
    real* wgtGrad = weight_->getWGrad()->getData();
    real* wgtVal = weight_->getW()->getData();
    real* biasGrad = hasBias ? biases_->getWGrad()->getData() : NULL;
    mkldnnBackwardFC(bs_,
                     ic_,
                     ih_,
                     iw_,
                     inGrad,
                     inVal,
                     oc_,
                     outGrad,
                     wgtGrad,
                     wgtVal,
                     biasGrad);
  }

  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
    if (hasBias) {
      biases_->getParameterPtr()->incUpdate(callback);
    }
  }
}
}  // namespace paddle
