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

// using namespace mkldnn;  // NOLINT
using mem = mkldnn::memory;  // NOLINT
typedef mem::format format;
typedef mkldnn::inner_product_forward fc_fwd;
typedef mkldnn::inner_product_backward_weights fc_bwdWgt;
typedef mkldnn::inner_product_backward_data fc_bwdData;

namespace paddle {

bool MkldnnLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  CHECK(FLAGS_use_mkldnn) << "MkldnnLayers only support use_mkldnn."
                          << "Please set WITH_MKLDNN=ON";
  // TODO(TJ): deivecId
  return Layer::init(layerMap, parameterMap);
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
  engine_ = CpuEngine::Instance().getEngine();

  mem::desc botMD = hasSpatial ? createMD({bs, ic, ih, iw}, format::nchw)
                               : createMD({bs, ic}, format::nc);
  mem::desc wgtMD = hasSpatial ? createMD({oc, ic, ih, iw}, format::oihw)
                               : createMD({oc, ic}, format::oi);
  mem::desc biasMD = biasData != NULL ? createMD({oc}, format::x)
                                      : createMD({}, format::format_undef);
  mem::desc topMD = createMD({bs, oc}, format::nc);

  mkldnn::prop_kind pk = mkldnn::prop_kind::forward;
  fc_fwd::desc fwdDesc = biasData != NULL
                             ? fc_fwd::desc(pk, botMD, wgtMD, biasMD, topMD)
                             : fc_fwd::desc(pk, botMD, wgtMD, topMD);
  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);

  mem bot = mem(mem::primitive_desc(botMD, engine_), botData);
  mem wgt = mem(mem::primitive_desc(wgtMD, engine_), wgtData);
  mem top = mem(mem::primitive_desc(topMD, engine_), topData);

  if (biasData != NULL) {
    mem bias = mem(mem::primitive_desc(biasMD, engine_), biasData);
    fwd_.reset(new fc_fwd(fwdPD, bot, wgt, bias, top));
  } else {
    fwd_.reset(new fc_fwd(fwdPD, bot, wgt, top));
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

  // just forward
  // update botdata
  stream_->submit(pipelineFwd_);
}

mem::desc MkldnnLayer::createMD(mem::dims dims,
                                mem::format fmt,
                                mem::data_type type) {
  // TODO(TJ): isFmtSuppoted(fmt)
  return mem::desc(dims, type, fmt);
}

}  // namespace paddle
