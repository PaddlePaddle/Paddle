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
  bool hasNoSpatial_ = ih_ == 1 && iw_ == 1;
  auto targetDim = wgtVal_->getDims();
  auto srcFmt = hasNoSpatial_ ? memory::format::io : memory::format::ihwo;
  wgtVal_->reorderDataFrom(wgtVal_, srcFmt, targetDim);
  hasInitedWgt_ = true;
}

void MKLDNNFcLayer::convertWeightsToPaddle() {
  CHECK(wgtVal_) << "should have been initialized";
  bool hasNoSpatial_ = ih_ == 1 && iw_ == 1;
  auto targetDim = wgtVal_->getDims();
  auto dstFmt = hasNoSpatial_ ? memory::format::io : memory::format::ihwo;
  wgtVal_->reorderDataTo(wgtVal_, dstFmt, targetDim);
}

void MKLDNNFcLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);

  CHECK_EQ(iLayerSize_, inputLayers_[0]->getSize());
  ic = iLayerSize_ / (ih * iw);
  CHECK_EQ(size_t(ic * ih * iw), iLayerSize_) << "not divisible";
  CHECK_EQ(size_t(oc), getSize());

  reshapeOutput(oh, ow);
  resizeOutput(bs, oc);

  printSizeInfo();
}

void MKLDNNFcLayer::resetFwd(std::vector<primitive>& pipeline,
                             MKLDNNMatrixPtr& in,
                             MKLDNNMatrixPtr& wgt,
                             MKLDNNMatrixPtr& bias,
                             MKLDNNMatrixPtr& out) {
  resetFwdBuffers(in, wgt, bias, out);

  resetFwdPD(fwdPD_, in, wgt, bias, out);

  resetFwdPipeline(pipeline, fwdPD_, in, wgt, bias, out);

  printValueFormatFlow();
}

void MKLDNNFcLayer::resetBwd(std::vector<primitive>& pipeline,
                             MKLDNNMatrixPtr& in,
                             MKLDNNMatrixPtr& wgt,
                             MKLDNNMatrixPtr& bias,
                             MKLDNNMatrixPtr& out) {
  std::shared_ptr<fc_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<fc_bwdData::primitive_desc> bwdDataPD;

  resetBwdBuffers(in, wgt, bias, out);

  resetBwdWgtPD(bwdWgtPD, wgt, bias, out);

  resetBwdDataPD(bwdDataPD, in, out);

  resetBwdPipeline(pipeline, bwdWgtPD, bwdDataPD, in, wgt, bias, out);

  printGradFormatFlow();
}

void MKLDNNFcLayer::updateInputData() {
  inVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
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

  resetWgtBiasValue(wgt, bias);

  resetOutValue(out);
}

void MKLDNNFcLayer::resetInValue(MKLDNNMatrixPtr& in) {
  if (inputIsOnlyMKLDNN()) {
    const MatrixPtr& dnnIn = getInputValue(0);
    in = std::dynamic_pointer_cast<MKLDNNMatrix>(dnnIn);
    CHECK(in) << "Input should be MKLDNNMatrix";
  } else {
    CHECK_EQ(getPrev(0)->getDeviceId(), CPU_DEVICE) << "Only support CPU yet";
    const MatrixPtr& cpuIn = getInputValue(0, CPU_DEVICE);
    in = MKLDNNMatrix::create(
        cpuIn, {bs_, ic_, ih_, iw_}, format::nchw, engine_);
  }
  in->downSpatial();
}

void MKLDNNFcLayer::resetWgtBiasValue(MKLDNNMatrixPtr& wgt,
                                      MKLDNNMatrixPtr& bias) {
  format wgtFmt = format::oihw;
  if (inVal_->getFormat() == format::nChw8c) {
    wgtFmt = format::oIhw8i;
  } else if (inVal_->getFormat() == format::nChw16c) {
    wgtFmt = format::oIhw16i;
  }
  wgt = MKLDNNMatrix::create(
      weight_->getW(), {oc_, ic_, ih_, iw_}, wgtFmt, engine_);
  wgt->downSpatial();
  VLOG(MKLDNN_FMTS) << "Weight value format: " << wgt->getFormat();

  bias = (biases_ && biases_->getW())
             ? MKLDNNMatrix::create(biases_->getW(), {oc_}, format::x, engine_)
             : nullptr;
}

void MKLDNNFcLayer::resetOutValue(MKLDNNMatrixPtr& out) {
  out = MKLDNNMatrix::create(output_.value, {bs_, oc_}, format::nc, engine_);
  if (!outputIsOnlyMKLDNN()) {
    // fc cpu output value do not need create convert
    // just share point
    getOutput(CPU_DEVICE).value->setData(out->getData());
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
  pipeline.clear();

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
  resetOutGrad(out);

  resetWgtBiasGrad(wgt, bias);

  resetInGrad(in);
}

void MKLDNNFcLayer::resetOutGrad(MKLDNNMatrixPtr& out) {
  // TODO(TJ): merge outgrad
  int device = outputIsOnlyMKLDNN() ? MKLDNN_DEVICE : CPU_DEVICE;
  output_.grad->setData(getOutput(device).grad->getData());
  // for MKLDNN device:
  // can not directly cast outputgrad to mkldnnmatrix,
  // since each layer can not write the inputgrad to mkldnn inputgrad.
  // So just create from matrix with outputvalue format.
  // for CPU device:
  // fc do not need to convert from cpu device since output is always nc format
  // only need create from cpu device
  CHECK(outVal_);
  out =
      MKLDNNMatrix::create(getOutput(device).grad, outVal_->getPrimitiveDesc());
}

void MKLDNNFcLayer::resetWgtBiasGrad(MKLDNNMatrixPtr& wgt,
                                     MKLDNNMatrixPtr& bias) {
  CHECK(wgtVal_);
  wgt = MKLDNNMatrix::create(weight_->getWGrad(), wgtVal_->getPrimitiveDesc());

  bias = nullptr;
  if (biasVal_ == nullptr) {
    return;
  }
  bias =
      MKLDNNMatrix::create(biases_->getWGrad(), biasVal_->getPrimitiveDesc());
}

void MKLDNNFcLayer::resetInGrad(MKLDNNMatrixPtr& in) {
  in = nullptr;
  const MatrixPtr& inGrad = inputLayers_[0]->getOutput().grad;
  if (inGrad == nullptr) {
    return;
  }
  // TODO(TJ): use outputMaps_ ways to get the inGrad_ when merge outgrad done
  CHECK(inVal_);
  in = MKLDNNMatrix::create(inGrad, inVal_->getPrimitiveDesc());
}

void MKLDNNFcLayer::resetBwdWgtPD(
    std::shared_ptr<fc_bwdWgt::primitive_desc>& pd,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(inVal_);
  fc_bwdWgt::desc bwdWgtDesc = bias ? fc_bwdWgt::desc(inVal_->getMemoryDesc(),
                                                      wgt->getMemoryDesc(),
                                                      bias->getMemoryDesc(),
                                                      out->getMemoryDesc())
                                    : fc_bwdWgt::desc(inVal_->getMemoryDesc(),
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
  pipeline.clear();
  CHECK(inVal_);
  if (bias) {
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD, *inVal_, *out, *wgt, *bias));
  } else {
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD, *inVal_, *out, *wgt));
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
