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

#include "MkldnnLayer.h"

using mem = mkldnn::memory;  // NOLINT
typedef mem::format format;
typedef mkldnn::inner_product_forward fc_fwd;
typedef mkldnn::inner_product_backward_weights fc_bwdWgt;
typedef mkldnn::inner_product_backward_data fc_bwdData;

namespace paddle {

bool MkldnnLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }

  CHECK(FLAGS_use_mkldnn) << "MkldnnLayers only support use_mkldnn."
                          << "Please set WITH_MKLDNN=ON "
                          << "and set use_mkldnn=True";
  stream_.reset(new MkldnnStream());
  engine_ = CpuEngine::Instance().getEngine();

  // TODO(TJ): deivecId
  return true;
}

void MkldnnLayer::resetForwardFC(int bs,
                                 int ic,
                                 int ih,
                                 int iw,
                                 real* botData,
                                 int oc,
                                 real* topData,
                                 real* wgtData,
                                 real* biasData) {
  bool hasSpatial = ih == 1 && iw == 1 ? false : true;
  mem::desc botMD = hasSpatial ? createMD({bs, ic, ih, iw}, format::nchw)
                               : createMD({bs, ic}, format::nc);
  mem::desc wgtMD = hasSpatial ? createMD({oc, ic, ih, iw}, format::oihw)
                               : createMD({oc, ic}, format::oi);
  mem::desc biasMD = biasData != NULL ? createMD({oc}, format::x)
                                      : createMD({}, format::format_undef);
  mem::desc topMD = createMD({bs, oc}, format::nc);

  mem::primitive_desc botPD = mem::primitive_desc(botMD, engine_);
  if (inVal_ && inVal_->get_primitive_desc() == botPD) {
    return;
  }

  inVal_.reset(new mem(botPD, botData));
  wgtVal_.reset(new mem(mem::primitive_desc(wgtMD, engine_), wgtData));
  outVal_.reset(new mem(mem::primitive_desc(topMD, engine_), topData));

  mkldnn::prop_kind pk = mkldnn::prop_kind::forward;
  fc_fwd::desc fwdDesc = biasData != NULL
                             ? fc_fwd::desc(pk, botMD, wgtMD, biasMD, topMD)
                             : fc_fwd::desc(pk, botMD, wgtMD, topMD);
  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);

  if (biasData != NULL) {
    biasVal_.reset(new mem(mem::primitive_desc(biasMD, engine_), biasData));
    fwd_.reset(new fc_fwd(fwdPD, *inVal_, *wgtVal_, *biasVal_, *outVal_));
  } else {
    fwd_.reset(new fc_fwd(fwdPD, *inVal_, *wgtVal_, *outVal_));
  }
  pipelineFwd_.clear();
  pipelineFwd_.push_back(*fwd_);
}

void MkldnnLayer::mkldnnForwardFC(int bs,
                                  int ic,
                                  int ih,
                                  int iw,
                                  real* botData,
                                  int oc,
                                  real* topData,
                                  real* wgtData,
                                  real* biasData) {
  // if input size changed, reset it
  resetForwardFC(bs, ic, ih, iw, botData, oc, topData, wgtData, biasData);

  this->convertWeightsFromPaddle();

  // update input, since the data might be changed if this is after data layer
  inVal_->set_data_handle(botData);

  // just forward
  stream_->submit(pipelineFwd_);
}

void MkldnnLayer::resetBackwardFC(int bs,
                                  int ic,
                                  int ih,
                                  int iw,
                                  real* botDiff,
                                  real* botData,
                                  int oc,
                                  real* topDiff,
                                  real* wgtDiff,
                                  real* wgtData,
                                  real* biasDiff) {
  bool hasSpatial = ih == 1 && iw == 1 ? false : true;

  // backward weight
  mem::desc botMD = hasSpatial ? createMD({bs, ic, ih, iw}, format::nchw)
                               : createMD({bs, ic}, format::nc);
  mem::desc wgtMD = hasSpatial ? createMD({oc, ic, ih, iw}, format::oihw)
                               : createMD({oc, ic}, format::oi);
  mem::desc topMD = createMD({bs, oc}, format::nc);
  mem::desc biasMD = biasDiff != NULL ? createMD({oc}, format::x)
                                      : createMD({}, format::format_undef);

  mem::primitive_desc topPD = mem::primitive_desc(botMD, engine_);
  if (outGrad_ && outGrad_->get_primitive_desc() == topPD) {
    return;
  }

  if (inVal_) {
    // update data
    inVal_->set_data_handle(botData);
  } else {
    inVal_.reset(new mem(mem::primitive_desc(botMD, engine_), botData));
  }
  wgtGrad_.reset(new mem(mem::primitive_desc(wgtMD, engine_), wgtDiff));
  outGrad_.reset(new mem(topPD, topDiff));

  fc_fwd::desc fwdDesc =
      fc_fwd::desc(mkldnn::prop_kind::forward, botMD, wgtMD, topMD);
  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);
  fc_bwdWgt::desc bwdWgtDesc =
      biasDiff != NULL ? fc_bwdWgt::desc(botMD, wgtMD, biasMD, topMD)
                       : fc_bwdWgt::desc(botMD, wgtMD, topMD);
  fc_bwdWgt::primitive_desc bwdWgtPD =
      fc_bwdWgt::primitive_desc(bwdWgtDesc, engine_, fwdPD);

  if (biasDiff != NULL) {
    biasGrad_.reset(new mem(mem::primitive_desc(biasMD, engine_), biasDiff));
    bwdWgt_.reset(
        new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_, *biasGrad_));
  } else {
    bwdWgt_.reset(new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_));
  }
  pipelineBwd_.clear();
  pipelineBwd_.push_back(*bwdWgt_);

  // backward data
  if (botDiff == NULL) {
    return;
  }

  fc_bwdData::desc bwdDataDesc = fc_bwdData::desc(botMD, wgtMD, topMD);
  fc_bwdData::primitive_desc bwdDataPD =
      fc_bwdData::primitive_desc(bwdDataDesc, engine_, fwdPD);
  inGrad_.reset(new mem(mem::primitive_desc(botMD, engine_), botDiff));
  if (wgtVal_) {
    // update data
    wgtVal_->set_data_handle(wgtData);
  } else {
    wgtVal_.reset(new mem(mem::primitive_desc(wgtMD, engine_), wgtData));
  }
  bwdData_.reset(new fc_bwdData(bwdDataPD, *outGrad_, *wgtVal_, *inGrad_));
  pipelineBwd_.push_back(*bwdData_);
}

void MkldnnLayer::mkldnnBackwardFC(int bs,
                                   int ic,
                                   int ih,
                                   int iw,
                                   real* botDiff,
                                   real* botData,
                                   int oc,
                                   real* topDiff,
                                   real* wgtDiff,
                                   real* wgtData,
                                   real* biasDiff) {
  // if input size changed, reset it
  resetBackwardFC(bs,
                  ic,
                  ih,
                  iw,
                  botDiff,
                  botData,
                  oc,
                  topDiff,
                  wgtDiff,
                  wgtData,
                  biasDiff);

  // update data
  outGrad_->set_data_handle(topDiff);

  stream_->submit(pipelineBwd_);
}

void MkldnnLayer::printSizeInfo() {
  VLOG(DNN_SIZES) << getName() << ": bs: " << bs_ << ", ic: " << ic_
                  << ", ih: " << ih_ << ", iw: " << iw_ << ", oc: " << oc_
                  << ", oh: " << oh_ << ", ow: " << ow_;
}

mem::desc MkldnnLayer::createMD(mem::dims dims,
                                mem::format fmt,
                                mem::data_type type) {
  // TODO(TJ): isFmtSuppoted(fmt)
  return mem::desc(dims, type, fmt);
}

}  // namespace paddle
