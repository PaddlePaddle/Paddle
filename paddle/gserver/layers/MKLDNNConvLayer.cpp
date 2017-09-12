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
  CHECK(dh_ == 1 && dw_ == 1) << "Only support dilation 1 yet";
  // check group setting
  CHECK_EQ((oc_ / gp_) * gp_, oc_) << "group is indivisible for oc";
  CHECK_EQ((ic_ / gp_) * gp_, ic_) << "group is indivisible for ic";

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
  // the paddle weight format is oihw or goihw
  auto targetDim = wgtVal_->getDims();
  auto srcFmt = (gp_ == 1) ? memory::format::oihw : memory::format::goihw;
  wgtVal_->reorderDataFrom(wgtVal_, srcFmt, targetDim);
  hasInitedWgt_ = true;
}

void MKLDNNConvLayer::convertWeightsToPaddle() {
  CHECK(wgtVal_) << "should have been initialized";
  auto targetDim = wgtVal_->getDims();
  auto dstFmt = (gp_ == 1) ? memory::format::oihw : memory::format::goihw;
  wgtVal_->reorderDataTo(wgtVal_, dstFmt, targetDim);
}

void MKLDNNConvLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);

  // cal output sizes
  // oc can not be changed
  int fh = (fh_ - 1) * dh_ + 1;
  int fw = (fw_ - 1) * dw_ + 1;
  oh = outputSize(ih, fh, ph_, sh_, caffeMode_);
  ow = outputSize(iw, fw, pw_, sw_, caffeMode_);

  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);

  printSizeInfo();
}

void MKLDNNConvLayer::resetFwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  pipeline.clear();
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
  prop_kind pk =
      passType_ == PASS_TEST ? prop_kind::forward : prop_kind::forward_training;
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
  fwdPD_.reset(new conv_fwd::primitive_desc(fwdDesc, engine_));

  // create mkldnn matrix
  const MatrixPtr& wgtVal = weight_->getW();
  const MatrixPtr& inVal = inputLayers_[0]->getOutput().value;
  const MatrixPtr& outVal = output_.value;
  wgt = MKLDNNMatrix::create(wgtVal, fwdPD_->weights_primitive_desc());
  in = MKLDNNMatrix::create(inVal, fwdPD_->src_primitive_desc());
  out = MKLDNNMatrix::create(outVal, fwdPD_->dst_primitive_desc());
  VLOG(MKLDNN_FMTS) << "Weight value format: " << wgtVal_->getFormat();
  if (hasBias) {
    const MatrixPtr& biasVal = biases_->getW();
    bias = MKLDNNMatrix::create(biasVal, biasDims, format::x, engine_);
    CHECK(bias->getPrimitiveDesc() == fwdPD_->bias_primitive_desc())
        << "bias primitive desc should always be equal";
  }

  // add reorder if input value do not match
  if (inputIsOnlyMKLDNN()) {
    MKLDNNMatrixPtr dnnIn = std::dynamic_pointer_cast<MKLDNNMatrix>(inVal);
    CHECK(dnnIn) << "Input should be MKLDNNMatrix";
    if (dnnIn->getPrimitiveDesc() != in->getPrimitiveDesc()) {
      CHECK_EQ(dnnIn->getFormat(), format::nc);
      CHECK(ih_ == 1 && iw_ == 1);
      dnnIn = MKLDNNMatrix::create(inVal, inDims, format::nchw, engine_);
      CHECK(dnnIn->getPrimitiveDesc() == in->getPrimitiveDesc());
    }
    in = dnnIn;
  } else {
    const MatrixPtr& cpuIn = getInputValue(0, CPU_DEVICE);
    cpuInVal_ = MKLDNNMatrix::create(cpuIn, inDims, format::nchw, engine_);
    if (cpuInVal_->getPrimitiveDesc() != in->getPrimitiveDesc()) {
      // create new mkldnn matrix
      in = MKLDNNMatrix::create(nullptr, fwdPD_->src_primitive_desc());
      cvtInVal_ = MKLDNNMatrix::createReorder(cpuInVal_, in);
      CHECK(cvtInVal_);
      pipeline.push_back(*cvtInVal_);
    } else {
      in = cpuInVal_;
    }
  }

  // add fwd handle
  if (hasBias) {
    fwd_.reset(new conv_fwd(*fwdPD_, *in, *wgt, *bias, *out));
  } else {
    fwd_.reset(new conv_fwd(*fwdPD_, *in, *wgt, *out));
  }
  pipeline.push_back(*fwd_);

  // change original output value from cpu matrix to mkldnn matrix
  output_.value = std::dynamic_pointer_cast<Matrix>(out);
  // add reorder if output value has cpu device and pd do not match
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).value;
    cpuOutVal_ = MKLDNNMatrix::create(cpuOut, outDims, format::nchw, engine_);
    if (cpuOutVal_->getPrimitiveDesc() != out->getPrimitiveDesc()) {
      cvtOutVal_ = MKLDNNMatrix::createReorder(out, cpuOutVal_);
      CHECK(cvtOutVal_);
      pipeline.push_back(*cvtOutVal_);
    } else {
      // share data
      cpuOut->setData(out->getData());
      cpuOutVal_ = out;
    }
  }

  printValueFormatFlow();
}

void MKLDNNConvLayer::resetBwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  pipeline.clear();
  bool hasBias = biases_ && biases_->getWGrad();

  /// backward weight
  CHECK(inVal_) << "Should have input value";
  CHECK(outVal_) << "Should have output value";
  CHECK(wgtVal_) << "Should have weight value";
  memory::dims wgtDims =
      (gp_ == 1) ? memory::dims{oc_, ic_, fh_, fw_}
                 : memory::dims{gp_, oc_ / gp_, ic_ / gp_, fh_, fw_};
  memory::dims strides = {sh_, sw_};
  memory::dims dilations = {dh_ - 1, dw_ - 1};
  memory::dims padding = {ph_, pw_};
  memory::dims padR = getPaddingR();

  // create backward handle
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  auto bwdWgtDesc =
      hasBias ? conv_bwdWgt::desc(algo,
                                  inVal_->getMemoryDesc(),
                                  MKLDNNMatrix::createMemoryDesc(wgtDims),
                                  biasVal_->getMemoryDesc(),
                                  outVal_->getMemoryDesc(),
                                  strides,
                                  padding,
                                  padR,
                                  padKind)
              : conv_bwdWgt::desc(algo,
                                  inVal_->getMemoryDesc(),
                                  MKLDNNMatrix::createMemoryDesc(wgtDims),
                                  outVal_->getMemoryDesc(),
                                  strides,
                                  padding,
                                  padR,
                                  padKind);

  auto bwdWgtPD = conv_bwdWgt::primitive_desc(bwdWgtDesc, engine_, *fwdPD_);
  CHECK(bwdWgtPD.src_primitive_desc() == inVal_->getPrimitiveDesc())
      << "primitive desc of in value should equal";
  CHECK(bwdWgtPD.diff_dst_primitive_desc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad should equal the out value";
  CHECK(bwdWgtPD.diff_weights_primitive_desc() == wgtVal_->getPrimitiveDesc())
      << "primitive desc of weight grad should equal the weight value";

  // create mkldnn matrix
  const MatrixPtr& wgtGrad = weight_->getWGrad();
  const MatrixPtr& outGrad = output_.grad;
  wgt = MKLDNNMatrix::create(wgtGrad, bwdWgtPD.diff_weights_primitive_desc());
  out = MKLDNNMatrix::create(outGrad, bwdWgtPD.diff_dst_primitive_desc());
  CHECK(wgt->getPrimitiveDesc() == wgtVal_->getPrimitiveDesc())
      << "primitive desc of weight grad and value should be equal";
  CHECK(out->getPrimitiveDesc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad and value should be equal";
  VLOG(MKLDNN_FMTS) << "Backward weight, weight grad format: "
                    << wgt->getFormat();
  if (hasBias) {
    const MatrixPtr& biasGrad = biases_->getWGrad();
    bias = MKLDNNMatrix::create(biasGrad, bwdWgtPD.diff_bias_primitive_desc());
    CHECK(bias->getPrimitiveDesc() == biasVal_->getPrimitiveDesc())
        << "primitive desc of bias grad should equal the bias value";
  }

  // TODO(TJ): merge outgrad
  // add reorder if has user output grad
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).grad;
    memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
    // same PrimitiveDesc with cpuInVal_
    CHECK(cpuOutVal_);
    cpuOutGrad_ = MKLDNNMatrix::create(cpuOut, cpuOutVal_->getPrimitiveDesc());
    if (cpuOutGrad_->getPrimitiveDesc() == out->getPrimitiveDesc()) {
      outGrad->setData(cpuOut->getData());
      out = cpuOutGrad_;
    } else {
      cvtOutGrad_ = MKLDNNMatrix::createReorder(cpuOutGrad_, out);
      CHECK(cvtOutGrad_);
      pipeline.push_back(*cvtOutGrad_);
    }
  }

  // add bwdWgt handle
  if (hasBias) {
    bwdWgt_.reset(new conv_bwdWgt(bwdWgtPD, *inVal_, *out, *wgt, *bias));
  } else {
    bwdWgt_.reset(new conv_bwdWgt(bwdWgtPD, *inVal_, *out, *wgt));
  }
  pipeline.push_back(*bwdWgt_);

  /// backward data
  const MatrixPtr& inGrad = inputLayers_[0]->getOutput().grad;
  if (inGrad == nullptr) {
    return;
  }

  auto bwdDataDesc = conv_bwdData::desc(algo,
                                        inVal_->getMemoryDesc(),
                                        MKLDNNMatrix::createMemoryDesc(wgtDims),
                                        out->getMemoryDesc(),
                                        strides,
                                        padding,
                                        padR,
                                        padKind);
  auto bwdDataPD = conv_bwdData::primitive_desc(bwdDataDesc, engine_, *fwdPD_);
  CHECK(bwdDataPD.diff_src_primitive_desc() == inVal_->getPrimitiveDesc())
      << "primitive desc of in grad should equal the in value";
  CHECK(bwdDataPD.diff_dst_primitive_desc() == out->getPrimitiveDesc())
      << "primitive desc of out grad should equal";

  // create mkldnn matrix inGrad_ and reorder if necessary
  // TODO(TJ): use outputMaps_ ways to get the inGrad_ when merge outgrad done
  in = MKLDNNMatrix::create(inGrad, bwdDataPD.diff_src_primitive_desc());
  cvtInGrad_ = nullptr;
  if (!inputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuIn = getInputGrad(0, CPU_DEVICE);
    // same PrimitiveDesc with cpuInVal_
    CHECK(cpuInVal_);
    cpuInGrad_ = MKLDNNMatrix::create(cpuIn, cpuInVal_->getPrimitiveDesc());
    if (cpuInGrad_->getPrimitiveDesc() != in->getPrimitiveDesc()) {
      const MatrixPtr& dnnIn = getInputGrad(0, MKLDNN_DEVICE);
      in = MKLDNNMatrix::create(dnnIn, in->getPrimitiveDesc());
      cvtInGrad_ = MKLDNNMatrix::createReorder(in, cpuInGrad_);
      CHECK(cvtInGrad_);
    } else {
      in = cpuInGrad_;
    }
  }

  // create new weight value for backward data, and reorder if necessary
  // since the primitive_desc would be different with wgtVal_
  if (bwdDataPD.weights_primitive_desc() != wgtVal_->getPrimitiveDesc()) {
    wgtValBwdData_ =
        MKLDNNMatrix::create(nullptr, bwdDataPD.weights_primitive_desc());
    cvtWgtVal_ = MKLDNNMatrix::createReorder(wgtVal_, wgtValBwdData_);
    CHECK(cvtWgtVal_);
    pipeline.push_back(*cvtWgtVal_);
  } else {
    wgtValBwdData_ = wgtVal_;
  }
  VLOG(MKLDNN_FMTS) << "Backward data, weight value format: "
                    << wgtValBwdData_->getFormat();

  // add bwdData handle
  CHECK(wgtValBwdData_) << "Should have weight memory";
  bwdData_.reset(new conv_bwdData(bwdDataPD, *out, *wgtValBwdData_, *in));
  pipeline.push_back(*bwdData_);

  // add ingrad reorder after bwdData
  if (cvtInGrad_) {
    pipeline.push_back(*cvtInGrad_);
  }

  printGradFormatFlow();
}

void MKLDNNConvLayer::updateInputData() {
  cpuInVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
}

void MKLDNNConvLayer::updateWeights(const UpdateCallback& callback) {
  weight_->getParameterPtr()->incUpdate(callback);
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
