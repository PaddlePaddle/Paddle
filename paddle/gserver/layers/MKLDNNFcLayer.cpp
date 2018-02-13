/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MKLDNNFcLayer);

bool MKLDNNFcLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  CHECK_EQ(inputLayers_.size(), 1UL) << "Only support one input layer yet";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK(!parameters_[0]->isSparse()) << "Do not support sparse yet";

  // output size, cat not be changed
  oc_ = getSize();
  oh_ = 1;
  ow_ = 1;
  ih_ = 1;
  iw_ = 1;

  // input size can not change in FC
  iLayerSize_ = inputLayers_[0]->getSize();
  CHECK_EQ(parameters_[0]->getSize(), iLayerSize_ * oc_);

  // create weight
  weight_ =
      std::unique_ptr<Weight>(new Weight(oc_, iLayerSize_, parameters_[0], 0));

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_, 0));
  }
  return true;
}

void MKLDNNFcLayer::convertWeightsFromPaddle() {
  if (hasInitedWgt_) {
    return;
  }

  CHECK(wgtVal_) << "should have been initialized";
  auto targetDim = wgtVal_->getDims();
  auto srcFmt = targetDim.size() == 2 ? format::io : format::ihwo;
  wgtVal_->reorderDataFrom(wgtVal_, srcFmt, targetDim);
  hasInitedWgt_ = true;
}

void MKLDNNFcLayer::convertWeightsToPaddle() {
  CHECK(wgtVal_) << "should have been initialized";
  auto targetDim = wgtVal_->getDims();
  auto dstFmt = targetDim.size() == 2 ? format::io : format::ihwo;
  wgtVal_->reorderDataTo(wgtVal_, dstFmt, targetDim);
}

void MKLDNNFcLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);

  CHECK_EQ(iLayerSize_, inputLayers_[0]->getSize());
  ic = iLayerSize_ / (ih * iw);
  CHECK_EQ(size_t(ic * ih * iw), iLayerSize_) << "not divisible";
  CHECK_EQ(size_t(oc), getSize());

  reshapeOutput(oh, ow);
  resizeOutput(bs, oc);
}

void MKLDNNFcLayer::resetFwd(std::vector<primitive>& pipeline,
                             std::vector<MKLDNNMatrixPtr>& inputs,
                             MKLDNNMatrixPtr& out) {
  resetFwdBuffers(inputs[0], wgtVal_, biasVal_, out);

  resetFwdPD(fwdPD_, inputs[0], wgtVal_, biasVal_, out);

  resetFwdPipeline(pipeline, fwdPD_, inputs[0], wgtVal_, biasVal_, out);
}

void MKLDNNFcLayer::resetBwd(std::vector<primitive>& pipeline,
                             std::vector<MKLDNNMatrixPtr>& inputs,
                             MKLDNNMatrixPtr& out) {
  std::shared_ptr<fc_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<fc_bwdData::primitive_desc> bwdDataPD;

  resetBwdBuffers(inputs[0], wgtGrad_, biasGrad_, out);

  resetBwdWgtPD(bwdWgtPD, wgtGrad_, biasGrad_, out);

  resetBwdDataPD(bwdDataPD, inputs[0], out);

  resetBwdPipeline(
      pipeline, bwdWgtPD, bwdDataPD, inputs[0], wgtGrad_, biasGrad_, out);
}

void MKLDNNFcLayer::updateWeights(const UpdateCallback& callback) {
  weight_->getParameterPtr()->incUpdate(callback);
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

void MKLDNNFcLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                    MKLDNNMatrixPtr& wgt,
                                    MKLDNNMatrixPtr& bias,
                                    MKLDNNMatrixPtr& out) {
  resetInValue(in);
  CHECK(in);
  in->downSpatial();

  auto outPD =
      MKLDNNMatrix::createPrimitiveDesc({bs_, oc_}, format::nc, engine_);
  resetOutValue(out, outPD);

  format wgtFmt = format::oihw;
  if (in->getFormat() == format::nChw8c) {
    wgtFmt = format::oIhw8i;
  } else if (in->getFormat() == format::nChw16c) {
    wgtFmt = format::oIhw16i;
  }
  auto wgtPD =
      MKLDNNMatrix::createPrimitiveDesc({oc_, ic_, ih_, iw_}, wgtFmt, engine_);
  resetWithMatrix(wgt, weight_->getW(), wgtPD);
  wgt->downSpatial();

  if (biases_ && biases_->getW()) {
    auto biasPD = MKLDNNMatrix::createPrimitiveDesc({oc_}, format::x, engine_);
    resetWithMatrix(bias, biases_->getW(), biasPD);
  } else {
    bias = nullptr;
  }
}

void MKLDNNFcLayer::resetFwdPD(std::shared_ptr<fc_fwd::primitive_desc>& pd,
                               MKLDNNMatrixPtr in,
                               MKLDNNMatrixPtr wgt,
                               MKLDNNMatrixPtr bias,
                               MKLDNNMatrixPtr out) {
  CHECK(in);
  CHECK(wgt);
  CHECK(out);
  prop_kind pk = prop_kind::forward;
  fc_fwd::desc fwdDesc = bias != nullptr ? fc_fwd::desc(pk,
                                                        in->getMemoryDesc(),
                                                        wgt->getMemoryDesc(),
                                                        bias->getMemoryDesc(),
                                                        out->getMemoryDesc())
                                         : fc_fwd::desc(pk,
                                                        in->getMemoryDesc(),
                                                        wgt->getMemoryDesc(),
                                                        out->getMemoryDesc());
  pd.reset(new fc_fwd::primitive_desc(fwdDesc, engine_));
}

void MKLDNNFcLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<fc_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  if (bias) {
    fwd_.reset(new fc_fwd(*pd, *in, *wgt, *bias, *out));
  } else {
    fwd_.reset(new fc_fwd(*pd, *in, *wgt, *out));
  }
  pipeline.push_back(*fwd_);
}

void MKLDNNFcLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                    MKLDNNMatrixPtr& wgt,
                                    MKLDNNMatrixPtr& bias,
                                    MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0] && outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  resetInGrad(in, inVals_[0]->getPrimitiveDesc());

  CHECK(wgtVal_);
  resetWithMatrix(wgt, weight_->getWGrad(), wgtVal_->getPrimitiveDesc());

  if (biasVal_) {
    resetWithMatrix(bias, biases_->getWGrad(), biasVal_->getPrimitiveDesc());
  } else {
    bias = nullptr;
  }
}

void MKLDNNFcLayer::resetBwdWgtPD(
    std::shared_ptr<fc_bwdWgt::primitive_desc>& pd,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0]);
  fc_bwdWgt::desc bwdWgtDesc =
      bias ? fc_bwdWgt::desc(inVals_[0]->getMemoryDesc(),
                             wgt->getMemoryDesc(),
                             bias->getMemoryDesc(),
                             out->getMemoryDesc())
           : fc_bwdWgt::desc(inVals_[0]->getMemoryDesc(),
                             wgt->getMemoryDesc(),
                             out->getMemoryDesc());
  pd.reset(new fc_bwdWgt::primitive_desc(bwdWgtDesc, engine_, *fwdPD_));
}

void MKLDNNFcLayer::resetBwdDataPD(
    std::shared_ptr<fc_bwdData::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {
  pd = nullptr;
  if (in == nullptr) {
    return;
  }
  CHECK(wgtVal_);
  fc_bwdData::desc bwdDataDesc = fc_bwdData::desc(
      in->getMemoryDesc(), wgtVal_->getMemoryDesc(), out->getMemoryDesc());
  pd.reset(new fc_bwdData::primitive_desc(bwdDataDesc, engine_, *fwdPD_));
}

void MKLDNNFcLayer::resetBwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
    std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0]);
  if (bias) {
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD, *inVals_[0], *out, *wgt, *bias));
  } else {
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD, *inVals_[0], *out, *wgt));
  }
  pipeline.push_back(*bwdWgt_);

  if (bwdDataPD == nullptr) {
    return;
  }
  CHECK(wgtVal_) << "Should have weight memory";
  bwdData_.reset(new fc_bwdData(*bwdDataPD, *out, *wgtVal_, *in));
  pipeline.push_back(*bwdData_);
}

}  // namespace paddle
