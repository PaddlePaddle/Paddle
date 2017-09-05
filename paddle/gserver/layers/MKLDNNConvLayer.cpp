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

#include "MKLDNNConvLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/utils/Logging.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;
typedef convolution_forward conv_fwd;
typedef convolution_backward_weights conv_bwdWgt;
typedef convolution_backward_data conv_bwdData;

namespace paddle {

REGISTER_LAYER(mkldnn_conv, MKLDNNConvLayer);

bool MKLDNNConvLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }
  CHECK_EQ(inputLayers_.size(), 1) << "Only support one input layer yet";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK(config_.shared_biases()) << "Only support shared biases yet";

  oc_ = config_.num_filters();
  const ConvConfig& conf = config_.inputs(0).conv_conf();
  ic_ = conf.channels();
  fw_ = conf.filter_size();
  fh_ = conf.filter_size_y();
  pw_ = conf.padding();
  ph_ = conf.padding_y();
  dw_ = conf.dilation();
  dh_ = conf.dilation_y();
  sw_ = conf.stride();
  sh_ = conf.stride_y();
  gp_ = conf.groups();
  oh_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  ow_ = conf.output_x();
  ih_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  iw_ = conf.img_size();
  caffeMode_ = conf.caffe_mode();
  CHECK(caffeMode_) << "Only support caffe mode yet";
  CHECK_EQ(gp_, 1) << "Only support group 1 yet";

  // create weight
  size_t height = oc_ / gp_;
  size_t width = ic_ * fh_ * fw_;
  CHECK_EQ(parameters_[0]->getSize(), height * width);
  weight_ =
      std::unique_ptr<Weight>(new Weight(height, width, parameters_[0], 0));

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

void MKLDNNConvLayer::convertWeightsFromPaddle() {
  if (hasInitedWgt_) {
    return;
  }

  CHECK(wgtVal_) << "should have been initialized";
  // TODO(TJ): check g>2 cpu format
  CHECK_EQ(gp_, 1);
  auto targetDim = wgtVal_->getDims();
  auto srcFmt = (gp_ == 1) ? memory::format::oihw : memory::format::goihw;
  wgtVal_->reorderDataFrom(wgtVal_, srcFmt, targetDim);
  hasInitedWgt_ = true;
}

void MKLDNNConvLayer::convertWeightsToPaddle() {
  CHECK(wgtVal_) << "should have been initialized";
  // TODO(TJ): check g>2 cpu format
  CHECK_EQ(gp_, 1);
  auto targetDim = wgtVal_->getDims();
  auto dstFmt = (gp_ == 1) ? memory::format::oihw : memory::format::goihw;
  wgtVal_->reorderDataTo(wgtVal_, dstFmt, targetDim);
}

void MKLDNNConvLayer::reshape() {
  reshapeInput();

  // cal output sizes
  // oc can not be changed
  int fh = (fh_ - 1) * dh_ + 1;
  int fw = (fw_ - 1) * dw_ + 1;
  oh_ = outputSize(ih_, fh, ph_, sh_, caffeMode_);
  ow_ = outputSize(iw_, fw, pw_, sw_, caffeMode_);

  reshapeOutput(oh_, ow_);
  resizeOutput(bs_, oc_ * oh_ * ow_);

  printSizeInfo();
}

void MKLDNNConvLayer::resetFwd() {
  pipelineFwd_.clear();
  bool hasBias = biases_ && biases_->getW();
  biasVal_ = nullptr;

  // dims for conv
  memory::dims inDims = memory::dims{bs_, ic_, ih_, iw_};
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  memory::dims wgtDims =
      (gp_ == 1) ? memory::dims{oc_, ic_, fh_, fw_}
                 : memory::dims{gp_, oc_ / gp_, ic_ / gp_, fh_, fw_};
  memory::dims biasDims = memory::dims{oc_};
  memory::dims strides = {sh_, sw_};
  // note: mkldnn dilation start from 0
  memory::dims dilations = {dh_ - 1, dw_ - 1};
  memory::dims padding = {ph_, pw_};
  memory::dims padR = getPaddingR();

  // create forward handle
  prop_kind pk = prop_kind::forward;
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  conv_fwd::desc fwdDesc =
      hasBias ? conv_fwd::desc(pk,
                               algo,
                               MKLDNNMatrix::createMemoryDesc(inDims),
                               MKLDNNMatrix::createMemoryDesc(wgtDims),
                               MKLDNNMatrix::createMemoryDesc(biasDims),
                               MKLDNNMatrix::createMemoryDesc(outDims),
                               strides,
                               dilations,
                               padding,
                               padR,
                               padKind)
              : conv_fwd::desc(pk,
                               algo,
                               MKLDNNMatrix::createMemoryDesc(inDims),
                               MKLDNNMatrix::createMemoryDesc(wgtDims),
                               MKLDNNMatrix::createMemoryDesc(outDims),
                               strides,
                               dilations,
                               padding,
                               padR,
                               padKind);
  conv_fwd::primitive_desc fwdPD = conv_fwd::primitive_desc(fwdDesc, engine_);

  // create mkldnn matrix
  const MatrixPtr& wgt = weight_->getW();
  const MatrixPtr& in = inputLayers_[0]->getOutput().value;
  const MatrixPtr& out = output_.value;
  wgtVal_ = MKLDNNMatrix::create(wgt, fwdPD.weights_primitive_desc());
  inVal_ = MKLDNNMatrix::create(in, fwdPD.src_primitive_desc());
  outVal_ = MKLDNNMatrix::create(out, fwdPD.dst_primitive_desc());
  VLOG(MKLDNN_FMTS) << "Weight value format: " << wgtVal_->getFormat();
  if (hasBias) {
    const MatrixPtr& bias = biases_->getW();
    biasVal_ = MKLDNNMatrix::create(bias, biasDims, format::x, engine_);
    CHECK(biasVal_->getPrimitiveDesc() == fwdPD.bias_primitive_desc())
        << "bias primitive desc should always be equal";
  }

  // add reorder if input value do not match
  if (inputIsOnlyMKLDNN()) {
    userInVal_ = std::dynamic_pointer_cast<MKLDNNMatrix>(in);
    CHECK(userInVal_) << "Input should be MKLDNNMatrix";
    if (userInVal_->getPrimitiveDesc() != inVal_->getPrimitiveDesc()) {
      CHECK_EQ(userInVal_->getFormat(), format::nc);
      CHECK(ih_ == 1 && iw_ == 1);
      userInVal_ = MKLDNNMatrix::create(in, inDims, format::nchw, engine_);
      CHECK(userInVal_->getPrimitiveDesc() == inVal_->getPrimitiveDesc());
    }
    inVal_ = userInVal_;
  } else {
    const MatrixPtr& cpuIn = getInputValue(0, CPU_DEVICE);
    userInVal_ = MKLDNNMatrix::create(cpuIn, inDims, format::nchw, engine_);
    if (userInVal_->getPrimitiveDesc() != inVal_->getPrimitiveDesc()) {
      // create new mkldnn matrix
      inVal_ = MKLDNNMatrix::create(nullptr, fwdPD.src_primitive_desc());
      cvtInVal_ = MKLDNNMatrix::createReorder(userInVal_, inVal_);
      CHECK(cvtInVal_);
      pipelineFwd_.push_back(*cvtInVal_);
    } else {
      inVal_ = userInVal_;
    }
  }

  // add fwd handle
  if (hasBias) {
    fwd_.reset(new conv_fwd(fwdPD, *inVal_, *wgtVal_, *biasVal_, *outVal_));
  } else {
    fwd_.reset(new conv_fwd(fwdPD, *inVal_, *wgtVal_, *outVal_));
  }
  pipelineFwd_.push_back(*fwd_);

  // change original output value from cpu matrix to mkldnn matrix
  output_.value = std::dynamic_pointer_cast<Matrix>(outVal_);
  // add reorder if output value has cpu device and pd do not match
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).value;
    userOutVal_ = MKLDNNMatrix::create(cpuOut, outDims, format::nchw, engine_);
    if (userOutVal_->getPrimitiveDesc() != outVal_->getPrimitiveDesc()) {
      cvtOutVal_ = MKLDNNMatrix::createReorder(outVal_, userOutVal_);
      CHECK(cvtOutVal_);
      pipelineFwd_.push_back(*cvtOutVal_);
    } else {
      // share data
      cpuOut->setData(outVal_->getData());
      userOutVal_ = outVal_;
    }
  }

  printValueFormatFlow();
}

void MKLDNNConvLayer::resetBwd() {
  pipelineBwd_.clear();
  bool hasBias = biases_ && biases_->getWGrad();

  /// backward weight
  CHECK(inVal_) << "Should have input value";
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  memory::dims wgtDims =
      (gp_ == 1) ? memory::dims{oc_, ic_, fh_, fw_}
                 : memory::dims{gp_, oc_ / gp_, ic_ / gp_, fh_, fw_};
  memory::dims biasDims = memory::dims{oc_};
  memory::dims strides = {sh_, sw_};
  // memory::dims dilations = {dh_ - 1, dw_ - 1};
  memory::dims padding = {ph_, pw_};
  memory::dims padR = getPaddingR();

  // create backward handle
  prop_kind pk = prop_kind::forward_training;
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  auto fwdDesc = hasBias
                     ? conv_fwd::desc(pk,
                                      algo,
                                      inVal_->getMemoryDesc(),
                                      MKLDNNMatrix::createMemoryDesc(wgtDims),
                                      MKLDNNMatrix::createMemoryDesc(biasDims),
                                      MKLDNNMatrix::createMemoryDesc(outDims),
                                      strides,
                                      // dilations,  // why?
                                      padding,
                                      padR,
                                      padKind)
                     : conv_fwd::desc(pk,
                                      algo,
                                      inVal_->getMemoryDesc(),
                                      MKLDNNMatrix::createMemoryDesc(wgtDims),
                                      MKLDNNMatrix::createMemoryDesc(outDims),
                                      strides,
                                      // dilations,  // why?
                                      padding,
                                      padR,
                                      padKind);
  auto bwdWgtDesc =
      hasBias ? conv_bwdWgt::desc(algo,
                                  inVal_->getMemoryDesc(),
                                  MKLDNNMatrix::createMemoryDesc(wgtDims),
                                  biasVal_->getMemoryDesc(),
                                  MKLDNNMatrix::createMemoryDesc(outDims),
                                  strides,
                                  padding,
                                  padR,
                                  padKind)
              : conv_bwdWgt::desc(algo,
                                  inVal_->getMemoryDesc(),
                                  MKLDNNMatrix::createMemoryDesc(wgtDims),
                                  MKLDNNMatrix::createMemoryDesc(outDims),
                                  strides,
                                  padding,
                                  padR,
                                  padKind);

  auto fwdPD = conv_fwd::primitive_desc(fwdDesc, engine_);
  auto bwdWgtPD = conv_bwdWgt::primitive_desc(bwdWgtDesc, engine_, fwdPD);
  CHECK(inVal_->getPrimitiveDesc() == bwdWgtPD.src_primitive_desc())
      << "primitive desc of in value should equal";

  // below are all internal memories
  const MatrixPtr& wgt = weight_->getWGrad();
  const MatrixPtr& out = output_.grad;
  wgtGrad_ = MKLDNNMatrix::create(wgt, bwdWgtPD.diff_weights_primitive_desc());
  outGrad_ = MKLDNNMatrix::create(out, bwdWgtPD.diff_dst_primitive_desc());
  CHECK(wgtGrad_->getPrimitiveDesc() == wgtVal_->getPrimitiveDesc())
      << "primitive desc of weight grad and value should be equal";
  CHECK(outGrad_->getPrimitiveDesc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad and value should be equal";

  VLOG(MKLDNN_FMTS) << "Backward weight, weight grad format: "
                    << wgtGrad_->getFormat();
  if (hasBias) {
    const MatrixPtr& bias = biases_->getWGrad();
    biasGrad_ = MKLDNNMatrix::create(bias, biasDims, format::x, engine_);
    CHECK(biasVal_->getPrimitiveDesc() == bwdWgtPD.diff_bias_primitive_desc())
        << "bias primitive desc should always be equal";
  }

  // TODO(TJ): merge outgrad
  // add reorder if has user output grad
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuGrad = getOutput(CPU_DEVICE).grad;
    userOutGrad_ =
        MKLDNNMatrix::create(cpuGrad, outDims, format::nchw, engine_);
    cvtOutGrad_ = MKLDNNMatrix::createReorder(userOutGrad_, outGrad_);
    if (cvtOutGrad_ == nullptr) {
      // TODO: check
      outGrad_->updateData(userOutGrad_->getData());
    } else {
      pipelineBwd_.push_back(*cvtOutGrad_);
    }
  }

  // add bwdWgt handle
  if (hasBias) {
    bwdWgt_.reset(
        new conv_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_, *biasGrad_));
  } else {
    bwdWgt_.reset(new conv_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_));
  }
  pipelineBwd_.push_back(*bwdWgt_);

  /// backward data
  const MatrixPtr& in = inputLayers_[0]->getOutput().grad;
  if (in == nullptr) {
    return;
  }

  if (inputIsOnlyMKLDNN()) {
    inGrad_ = MKLDNNMatrix::create(in, inVal_->getPrimitiveDesc());
  } else {
    // TODO(TJ): use outputMaps_ ways to get the inGrad_ when merge outgrad done
    const MatrixPtr& dnnIn = inputLayers_[0]->getOutput(MKLDNN_DEVICE).grad;
    inGrad_ = MKLDNNMatrix::create(dnnIn, inVal_->getPrimitiveDesc());
  }

  auto bwdDataFwdDesc = conv_fwd::desc(pk,
                                       algo,
                                       inGrad_->getMemoryDesc(),
                                       MKLDNNMatrix::createMemoryDesc(wgtDims),
                                       outGrad_->getMemoryDesc(),
                                       strides,
                                       // dilations,  // why?
                                       padding,
                                       padR,
                                       padKind);
  auto bwdDataFwdPD = conv_fwd::primitive_desc(bwdDataFwdDesc, engine_);
  auto bwdDataDesc = conv_bwdData::desc(algo,
                                        inGrad_->getMemoryDesc(),
                                        MKLDNNMatrix::createMemoryDesc(wgtDims),
                                        outGrad_->getMemoryDesc(),
                                        strides,
                                        padding,
                                        padR,
                                        padKind);
  auto bwdDataPD =
      conv_bwdData::primitive_desc(bwdDataDesc, engine_, bwdDataFwdPD);
  CHECK(inGrad_->getPrimitiveDesc() == bwdDataPD.diff_src_primitive_desc());
  CHECK(outGrad_->getPrimitiveDesc() == bwdDataPD.diff_dst_primitive_desc());

  // create reorder weight, since the format would be differ with bwdWgt
  int wgtFmt = bwdDataPD.weights_primitive_desc().desc().data.format;
  if (wgtFmt != wgtVal_->getFormat()) {
    wgtValBwdData_ =
        MKLDNNMatrix::create(nullptr, bwdDataPD.weights_primitive_desc());
    cvtWgtVal_ = MKLDNNMatrix::createReorder(wgtVal_, wgtValBwdData_);
    if (cvtWgtVal_ != nullptr) {
      pipelineBwd_.push_back(*cvtWgtVal_);
    } else {
      wgtValBwdData_ = wgtVal_;
    }
  } else {
    wgtValBwdData_ = wgtVal_;
  }
  VLOG(MKLDNN_FMTS) << "Backward data, weight value format: "
                    << wgtValBwdData_->getFormat();

  CHECK(wgtValBwdData_) << "Should have weight memory";
  bwdData_.reset(
      new conv_bwdData(bwdDataPD, *outGrad_, *wgtValBwdData_, *inGrad_));
  pipelineBwd_.push_back(*bwdData_);

  // add reorder of in grad
  if (!inputIsOnlyMKLDNN()) {
    CHECK(userInVal_);
    userInGrad_ = MKLDNNMatrix::create(in, userInVal_->getPrimitiveDesc());
    cvtInGrad_ = MKLDNNMatrix::createReorder(inGrad_, userInGrad_);
    if (cvtInGrad_ != nullptr) {
      pipelineBwd_.push_back(*cvtInGrad_);
    } else {
      // TODO: check
      inGrad_->updateData(userInGrad_->getData());
    }
  }
  printGradFormatFlow();
}

void MKLDNNConvLayer::updateInputData() {
  if (inputLayers_[0]->getType() != "data") {
    return;
  }
  real* iData = getInputValue(0, CPU_DEVICE)->getData();
  userInVal_->updateData(iData);
}

void MKLDNNConvLayer::updateWeights(const UpdateCallback& callback) {
  weight_->getParameterPtr()->incUpdate(callback);
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

void MKLDNNConvLayer::printSizeInfo() {
  MKLDNNLayer::printSizeInfo();
  VLOG(MKLDNN_SIZES) << getName() << ": fh: " << fh_ << ", fw: " << fw_
                     << ": ph: " << ph_ << ", pw: " << pw_ << ", sh: " << sh_
                     << ", sw: " << sw_ << ", dh: " << dh_ << ", dw: " << dw_;
}

memory::dims MKLDNNConvLayer::getPaddingR() const {
  memory::dims padR = {ph_, pw_};
  for (int i = 0; i < 2; ++i) {
    if ((ih_ - ((fh_ - 1) * dh_ + 1) + ph_ + padR[0]) / sh_ + 1 != oh_) {
      ++padR[0];
    }
    if ((iw_ - ((fw_ - 1) * dw_ + 1) + pw_ + padR[1]) / sw_ + 1 != ow_) {
      ++padR[1];
    }
  }
  return padR;
}

}  // namespace paddle
