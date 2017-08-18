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

#include "MKLDNNFcLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;
typedef inner_product_forward fc_fwd;
typedef inner_product_backward_weights fc_bwdWgt;
typedef inner_product_backward_data fc_bwdData;

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MKLDNNFcLayer);

bool MKLDNNFcLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  CHECK_EQ(inputLayers_.size(), 1) << "Only support one input layer yet";
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

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

void MKLDNNFcLayer::convertWeightsFromPaddle() {
  if (hasInitedWgt_) {
    return;
  }

  // TODO(TJ): dst format should get from wgtVal_
  int dstFmt = PARAM_FORMAT_MKLDNN_OI;
  int srcFmt = weight_->getParameterPtr()->getHeaderFormat();
  if (srcFmt == dstFmt) {
    return;
  }

  // The weight_ is transposed from initial paddle weight
  MatrixPtr paddleWgt = Matrix::create(
      weight_->getW()->getData(), iLayerSize_, oc_, false, false);

  // TODO(TJ): remove this print when do not need differ weights
  std::ostringstream ostr;
  paddleWgt->print(ostr);
  VLOG(MKLDNN_ALL) << "Initial Weight from paddle: " << std::endl << ostr.str();

  // The mkldnn weight is transposed from initial paddle matrix
  MatrixPtr paddleWgtT;
  paddleWgt->transpose(paddleWgtT, true);
  weight_->getW()->copyFrom(*paddleWgtT);
  weight_->getParameterPtr()->setHeaderFormat(dstFmt);
  hasInitedWgt_ = true;
}

void MKLDNNFcLayer::convertWeightsToPaddle() {
  MatrixPtr dnnWgt = weight_->getW();
  MatrixPtr paddleWgt;
  dnnWgt->transpose(paddleWgt, true);

  // copy paddle weight and override on weight_
  MatrixPtr dnnWgtT = Matrix::create(
      dnnWgt->getData(), dnnWgt->getWidth(), dnnWgt->getHeight(), false, false);
  dnnWgtT->copyFrom(*paddleWgt);
}

void MKLDNNFcLayer::reshape() {
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
  hasSpatial_ = true;
  if (ih_ == 1 && iw_ == 1) {
    hasSpatial_ = false;
  }
  CHECK_EQ(iLayerSize_, inputLayers_[0]->getSize());
  ic_ = iLayerSize_ / (ih_ * iw_);
  CHECK_EQ(size_t(ic_ * ih_ * iw_), iLayerSize_) << "not divisible";
  CHECK_EQ(size_t(oc_), getSize());
  printSizeInfo();

  // reset output
  output_.setFrameHeight(oh_);
  output_.setFrameWidth(ow_);
  resetOutput(bs_, oc_);

  // reset mkldnn forward
  resetFwd();
  needResetBwd_ = true;

  convertWeightsFromPaddle();
}

void MKLDNNFcLayer::resetFwd() {
  bool hasBias = biases_ && biases_->getW();
  real* iData = getInputValue(0)->getData();
  real* oData = getOutputValue()->getData();
  real* wData = weight_->getW()->getData();
  real* bData = hasBias ? biases_->getW()->getData() : NULL;

  // TODO(TJ): below create should be covered in MkldnnMatrix
  // create memory desc
  memory::desc iMD = hasSpatial_ ? createMD({bs_, ic_, ih_, iw_}, format::nchw)
                                 : createMD({bs_, ic_}, format::nc);
  memory::desc wMD = hasSpatial_ ? createMD({oc_, ic_, ih_, iw_}, format::oihw)
                                 : createMD({oc_, ic_}, format::oi);
  memory::desc bMD = bData != NULL ? createMD({oc_}, format::x)
                                   : createMD({}, format::format_undef);
  memory::desc oMD = createMD({bs_, oc_}, format::nc);

  // create memory primitive desc and memory self
  inVal_.reset(new memory(memory::primitive_desc(iMD, engine_), iData));
  wgtVal_.reset(new memory(memory::primitive_desc(wMD, engine_), wData));
  outVal_.reset(new memory(memory::primitive_desc(oMD, engine_), oData));

  prop_kind pk = prop_kind::forward;
  fc_fwd::desc fwdDesc = bData != NULL ? fc_fwd::desc(pk, iMD, wMD, bMD, oMD)
                                       : fc_fwd::desc(pk, iMD, wMD, oMD);
  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);

  if (bData != NULL) {
    biasVal_.reset(new memory(memory::primitive_desc(bMD, engine_), bData));
    fwd_.reset(new fc_fwd(fwdPD, *inVal_, *wgtVal_, *biasVal_, *outVal_));
  } else {
    fwd_.reset(new fc_fwd(fwdPD, *inVal_, *wgtVal_, *outVal_));
  }
  pipelineFwd_.clear();
  pipelineFwd_.push_back(*fwd_);
}

void MKLDNNFcLayer::resetBwd() {
  if (!needResetBwd_) {
    return;
  }
  needResetBwd_ = false;

  bool hasBias = biases_ && biases_->getWGrad();
  real* iData = getInputValue(0)->getData();
  real* iDiff = getInputGrad(0) != nullptr ? getInputGrad(0)->getData() : NULL;
  real* oDiff = getOutputGrad()->getData();
  real* wDiff = weight_->getWGrad()->getData();
  real* bDiff = hasBias ? biases_->getWGrad()->getData() : NULL;

  /// backward weight
  // create memory desc for backward memory
  memory::desc iMD = hasSpatial_ ? createMD({bs_, ic_, ih_, iw_}, format::nchw)
                                 : createMD({bs_, ic_}, format::nc);
  memory::desc wMD = hasSpatial_ ? createMD({oc_, ic_, ih_, iw_}, format::oihw)
                                 : createMD({oc_, ic_}, format::oi);
  memory::desc oMD = createMD({bs_, oc_}, format::nc);
  memory::desc bMD = bDiff != NULL ? createMD({oc_}, format::x)
                                   : createMD({}, format::format_undef);

  if (inVal_) {
    // update data
    inVal_->set_data_handle(iData);
  } else {
    inVal_.reset(new memory(memory::primitive_desc(iMD, engine_), iData));
  }

  // create memory primitive desc and memory self
  wgtGrad_.reset(new memory(memory::primitive_desc(wMD, engine_), wDiff));
  outGrad_.reset(new memory(memory::primitive_desc(oMD, engine_), oDiff));

  fc_fwd::desc fwdDesc = fc_fwd::desc(prop_kind::forward, iMD, wMD, oMD);
  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);
  fc_bwdWgt::desc bwdWgtDesc = bDiff != NULL
                                   ? fc_bwdWgt::desc(iMD, wMD, bMD, oMD)
                                   : fc_bwdWgt::desc(iMD, wMD, oMD);
  fc_bwdWgt::primitive_desc bwdWgtPD =
      fc_bwdWgt::primitive_desc(bwdWgtDesc, engine_, fwdPD);

  if (bDiff != NULL) {
    biasGrad_.reset(new memory(memory::primitive_desc(bMD, engine_), bDiff));
    bwdWgt_.reset(
        new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_, *biasGrad_));
  } else {
    bwdWgt_.reset(new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_));
  }
  pipelineBwd_.clear();
  pipelineBwd_.push_back(*bwdWgt_);

  /// backward data
  if (iDiff == NULL) {
    return;
  }
  fc_bwdData::desc bwdDataDesc = fc_bwdData::desc(iMD, wMD, oMD);
  fc_bwdData::primitive_desc bwdDataPD =
      fc_bwdData::primitive_desc(bwdDataDesc, engine_, fwdPD);
  inGrad_.reset(new memory(memory::primitive_desc(iMD, engine_), iDiff));
  CHECK(wgtVal_) << "Should have weight memory";
  bwdData_.reset(new fc_bwdData(bwdDataPD, *outGrad_, *wgtVal_, *inGrad_));
  pipelineBwd_.push_back(*bwdData_);
}

void MKLDNNFcLayer::forward(PassType passType) {
  Layer::forward(passType);
  reshape();

  {
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());

    // update input data
    // since it might be changed if this is after data layer
    real* iData = getInputValue(0)->getData();
    inVal_->set_data_handle(iData);

    // just submit forward pipeline
    stream_->submit(pipelineFwd_);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwActTimer", getName().c_str());
    forwardActivation();
  }
}

void MKLDNNFcLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpActTimer", getName().c_str());
    backwardActivation();
  }

  {
    REGISTER_TIMER_INFO("mkldnn_bwdTimer", getName().c_str());
    resetBwd();

    // update diff
    real* oDiff = getOutputGrad()->getData();
    outGrad_->set_data_handle(oDiff);

    // just sumbmit backward pipeline
    stream_->submit(pipelineBwd_);
  }

  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
    if (biases_ && biases_->getWGrad()) {
      biases_->getParameterPtr()->incUpdate(callback);
    }
  }
}
}  // namespace paddle
