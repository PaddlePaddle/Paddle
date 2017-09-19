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
  oh_ = conf.output_y();
  ow_ = conf.output_x();
  ih_ = conf.img_size_y();
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
  resetFwdPD(fwdPD_);

  resetFwdBuffers(fwdPD_, in, wgt, bias, out);

  resetFwdPipeline(pipeline, fwdPD_, in, wgt, bias, out);

  printValueFormatFlow();
}

void MKLDNNConvLayer::resetBwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  std::shared_ptr<conv_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<conv_bwdData::primitive_desc> bwdDataPD;

  resetBwdWgtPD(bwdWgtPD);

  resetBwdDataPD(bwdDataPD);

  resetBwdBuffers(bwdWgtPD, bwdDataPD, in, wgt, bias, out);

  resetBwdPipeline(pipeline, bwdWgtPD, bwdDataPD, in, wgt, bias, out);

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

void MKLDNNConvLayer::loadConvSettings(memory::dims& wgt,
                                       memory::dims& bias,
                                       memory::dims& stride,
                                       memory::dims& dilation,
                                       memory::dims& padL,
                                       memory::dims& padR) {
  wgt = (gp_ == 1) ? memory::dims{oc_, ic_, fh_, fw_}
                   : memory::dims{gp_, oc_ / gp_, ic_ / gp_, fh_, fw_};
  bias = memory::dims{oc_};
  stride = memory::dims{sh_, sw_};
  padL = memory::dims{ph_, pw_};
  padR = getPaddingR();
  // note: mkldnn dilation start from 0
  dilation = memory::dims{dh_ - 1, dw_ - 1};
}

void MKLDNNConvLayer::resetFwdPD(
    std::shared_ptr<conv_fwd::primitive_desc>& pd) {
  // dims for conv
  memory::dims inDims = memory::dims{bs_, ic_, ih_, iw_};
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  memory::dims wgtDims, biasDims, strides, dilations, padL, padR;
  loadConvSettings(wgtDims, biasDims, strides, dilations, padL, padR);

  prop_kind pk = passType_ == PASS_TEST ? prop_kind::forward_scoring
                                        : prop_kind::forward_training;
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  conv_fwd::desc fwdDesc =
      biases_ && biases_->getW()
          ? conv_fwd::desc(pk,
                           algo,
                           MKLDNNMatrix::createMemoryDesc(inDims),
                           MKLDNNMatrix::createMemoryDesc(wgtDims),
                           MKLDNNMatrix::createMemoryDesc(biasDims),
                           MKLDNNMatrix::createMemoryDesc(outDims),
                           strides,
                           dilations,
                           padL,
                           padR,
                           padKind)
          : conv_fwd::desc(pk,
                           algo,
                           MKLDNNMatrix::createMemoryDesc(inDims),
                           MKLDNNMatrix::createMemoryDesc(wgtDims),
                           MKLDNNMatrix::createMemoryDesc(outDims),
                           strides,
                           dilations,
                           padL,
                           padR,
                           padKind);
  pd.reset(new conv_fwd::primitive_desc(fwdDesc, engine_));
}

void MKLDNNConvLayer::resetFwdBuffers(
    std::shared_ptr<conv_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(pd);
  resetInValue(pd, in);

  resetWgtBiasValue(pd, wgt, bias);

  resetOutValue(pd, out);
}

void MKLDNNConvLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<conv_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  pipeline.clear();

  if (cvtInVal_) {
    pipeline.push_back(*cvtInVal_);
  }

  if (bias) {
    fwd_.reset(new conv_fwd(*pd, *in, *wgt, *bias, *out));
  } else {
    fwd_.reset(new conv_fwd(*pd, *in, *wgt, *out));
  }
  pipeline.push_back(*fwd_);

  if (cvtOutVal_) {
    pipeline.push_back(*cvtOutVal_);
  }
}

void MKLDNNConvLayer::resetInValue(
    std::shared_ptr<conv_fwd::primitive_desc>& pd, MKLDNNMatrixPtr& in) {
  const MatrixPtr& inMat = inputLayers_[0]->getOutput().value;
  in = MKLDNNMatrix::create(inMat, pd->src_primitive_desc());

  // create buffer and reorder if input value do not match
  cpuInVal_ = nullptr;
  cvtInVal_ = nullptr;
  if (inputIsOnlyMKLDNN()) {
    MKLDNNMatrixPtr dnnIn = std::dynamic_pointer_cast<MKLDNNMatrix>(inMat);
    CHECK(dnnIn) << "Input should be MKLDNNMatrix";
    if (dnnIn->getPrimitiveDesc() != in->getPrimitiveDesc()) {
      CHECK_EQ(dnnIn->getFormat(), format::nc);
      CHECK(ih_ == 1 && iw_ == 1) << "when input is nc format";
      // create a new one with nchw format and same data
      memory::dims inDims = memory::dims{bs_, ic_, 1, 1};
      dnnIn = MKLDNNMatrix::create(inMat, inDims, format::nchw, engine_);
      CHECK(dnnIn->getPrimitiveDesc() == in->getPrimitiveDesc());
    }
    in = dnnIn;
  } else {
    const MatrixPtr& cpuIn = getInputValue(0, CPU_DEVICE);
    memory::dims inDims = memory::dims{bs_, ic_, ih_, iw_};
    cpuInVal_ = MKLDNNMatrix::create(cpuIn, inDims, format::nchw, engine_);
    if (cpuInVal_->getPrimitiveDesc() != in->getPrimitiveDesc()) {
      // create new mkldnn matrix
      in = MKLDNNMatrix::create(nullptr, pd->src_primitive_desc());
      cvtInVal_ = MKLDNNMatrix::createReorder(cpuInVal_, in);
      CHECK(cvtInVal_) << "should not be emptry";
    } else {
      in = cpuInVal_;
    }
  }
}

void MKLDNNConvLayer::resetWgtBiasValue(
    std::shared_ptr<conv_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias) {
  wgt = MKLDNNMatrix::create(weight_->getW(), pd->weights_primitive_desc());
  VLOG(MKLDNN_FMTS) << "Weight value format: " << wgt->getFormat();

  bias = (biases_ && biases_->getW())
             ? MKLDNNMatrix::create(biases_->getW(), pd->bias_primitive_desc())
             : nullptr;
}

void MKLDNNConvLayer::resetOutValue(
    std::shared_ptr<conv_fwd::primitive_desc>& pd, MKLDNNMatrixPtr& out) {
  out = MKLDNNMatrix::create(output_.value, pd->dst_primitive_desc());

  // change original output value from cpu matrix to mkldnn matrix
  output_.value = std::dynamic_pointer_cast<Matrix>(out);

  // create reorder if output value has cpu device and pd do not match
  cpuOutVal_ = nullptr;
  cpuOutVal_ = nullptr;
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).value;
    memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
    cpuOutVal_ = MKLDNNMatrix::create(cpuOut, outDims, format::nchw, engine_);
    if (cpuOutVal_->getPrimitiveDesc() != out->getPrimitiveDesc()) {
      cvtOutVal_ = MKLDNNMatrix::createReorder(out, cpuOutVal_);
      CHECK(cvtOutVal_) << "should not be emptry";
    } else {
      // CPU output share the same data of MKLDNN output
      cpuOut->setData(out->getData());
      cpuOutVal_ = out;
    }
  }
}

void MKLDNNConvLayer::resetBwdWgtPD(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& pd) {
  memory::dims wgtDims, biasDims, strides, dilations, padL, padR;
  loadConvSettings(wgtDims, biasDims, strides, dilations, padL, padR);

  // create backward weight using input, output and weight value memory desc
  CHECK(inVal_) << "Should have input value";
  CHECK(outVal_) << "Should have output value";
  CHECK(wgtVal_) << "Should have weight value";
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  auto bwdWgtDesc = biasVal_ != nullptr
                        ? conv_bwdWgt::desc(algo,
                                            inVal_->getMemoryDesc(),
                                            wgtVal_->getMemoryDesc(),
                                            biasVal_->getMemoryDesc(),
                                            outVal_->getMemoryDesc(),
                                            strides,
                                            padL,
                                            padR,
                                            padKind)
                        : conv_bwdWgt::desc(algo,
                                            inVal_->getMemoryDesc(),
                                            wgtVal_->getMemoryDesc(),
                                            outVal_->getMemoryDesc(),
                                            strides,
                                            padL,
                                            padR,
                                            padKind);
  pd.reset(new conv_bwdWgt::primitive_desc(bwdWgtDesc, engine_, *fwdPD_));
  CHECK(pd->src_primitive_desc() == inVal_->getPrimitiveDesc())
      << "primitive desc of in value should equal";
  CHECK(pd->diff_dst_primitive_desc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad should equal the out value";
  CHECK(pd->diff_weights_primitive_desc() == wgtVal_->getPrimitiveDesc())
      << "primitive desc of weight grad should equal the weight value";
}

void MKLDNNConvLayer::resetBwdDataPD(
    std::shared_ptr<conv_bwdData::primitive_desc>& pd) {
  pd = nullptr;
  if (inputLayers_[0]->getOutput().grad == nullptr) {
    return;
  }

  memory::dims wgtDims, biasDims, strides, dilations, padL, padR;
  loadConvSettings(wgtDims, biasDims, strides, dilations, padL, padR);
  CHECK(inVal_) << "Should have input value";
  CHECK(outVal_) << "Should have output value";
  // create backward data using input and output value memory desc
  // but using weight memory desc with any format
  auto bwdDataDesc = conv_bwdData::desc(algorithm::convolution_direct,
                                        inVal_->getMemoryDesc(),
                                        MKLDNNMatrix::createMemoryDesc(wgtDims),
                                        outVal_->getMemoryDesc(),
                                        strides,
                                        padL,
                                        padR,
                                        padding_kind::zero);
  pd.reset(new conv_bwdData::primitive_desc(bwdDataDesc, engine_, *fwdPD_));
  CHECK(pd->diff_src_primitive_desc() == inVal_->getPrimitiveDesc())
      << "primitive desc of in grad should equal the in value";
  CHECK(pd->diff_dst_primitive_desc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad should equal";
}

void MKLDNNConvLayer::resetBwdBuffers(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
    std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(wgtPD);
  resetOutGrad(wgtPD, out);

  resetWgtBiasGrad(wgtPD, wgt, bias);

  resetInGrad(dataPD, in);

  resetWgtValBwdData(dataPD, wgtValBwdData_);
}

void MKLDNNConvLayer::resetBwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
    std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  pipeline.clear();

  if (cvtOutGrad_) {
    pipeline.push_back(*cvtOutGrad_);
  }

  // add bwdWgt handle
  if (bias) {
    bwdWgt_.reset(new conv_bwdWgt(*wgtPD, *inVal_, *out, *wgt, *bias));
  } else {
    bwdWgt_.reset(new conv_bwdWgt(*wgtPD, *inVal_, *out, *wgt));
  }
  pipeline.push_back(*bwdWgt_);

  if (dataPD == nullptr) {
    return;
  }

  if (cvtWgtVal_) {
    pipeline.push_back(*cvtWgtVal_);
  }

  // add bwdData handle
  CHECK(wgtValBwdData_) << "Should have weight memory";
  bwdData_.reset(new conv_bwdData(*dataPD, *out, *wgtValBwdData_, *in));
  pipeline.push_back(*bwdData_);

  if (cvtInGrad_) {
    pipeline.push_back(*cvtInGrad_);
  }
}

void MKLDNNConvLayer::resetOutGrad(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD, MKLDNNMatrixPtr& out) {
  const MatrixPtr& outMat = output_.grad;
  out = MKLDNNMatrix::create(outMat, wgtPD->diff_dst_primitive_desc());
  CHECK(outVal_ != nullptr &&
        out->getPrimitiveDesc() == outVal_->getPrimitiveDesc())
      << "primitive desc of out grad and value should be equal";

  // TODO(TJ): merge outgrad
  // create reorder if has output grad does not match
  cpuOutGrad_ = nullptr;
  cvtOutGrad_ = nullptr;
  if (!outputIsOnlyMKLDNN()) {
    const MatrixPtr& cpuOut = getOutput(CPU_DEVICE).grad;
    // same PrimitiveDesc with cpuInVal_
    CHECK(cpuOutVal_);
    cpuOutGrad_ = MKLDNNMatrix::create(cpuOut, cpuOutVal_->getPrimitiveDesc());
    if (cpuOutGrad_->getPrimitiveDesc() == out->getPrimitiveDesc()) {
      outMat->setData(cpuOut->getData());
      out = cpuOutGrad_;
    } else {
      cvtOutGrad_ = MKLDNNMatrix::createReorder(cpuOutGrad_, out);
      CHECK(cvtOutGrad_);
    }
  }
}

void MKLDNNConvLayer::resetWgtBiasGrad(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias) {
  wgt = MKLDNNMatrix::create(weight_->getWGrad(),
                             wgtPD->diff_weights_primitive_desc());
  CHECK(nullptr != wgtVal_ &&
        wgt->getPrimitiveDesc() == wgtVal_->getPrimitiveDesc())
      << "primitive desc of weight grad and value should be equal";
  VLOG(MKLDNN_FMTS) << "weight grad format: " << wgt->getFormat();

  bias = nullptr;
  if (biasVal_ == nullptr) {
    return;
  }
  bias = MKLDNNMatrix::create(biases_->getWGrad(),
                              wgtPD->diff_bias_primitive_desc());
  CHECK(bias->getPrimitiveDesc() == biasVal_->getPrimitiveDesc())
      << "primitive desc of bias grad should equal the bias value";
}

void MKLDNNConvLayer::resetInGrad(
    std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
    MKLDNNMatrixPtr& in) {
  if (dataPD == nullptr) {
    return;
  }

  // TODO(TJ): use outputMaps_ ways to get the inGrad_ when merge outgrad done
  in = MKLDNNMatrix::create(inputLayers_[0]->getOutput().grad,
                            dataPD->diff_src_primitive_desc());
  CHECK(nullptr != inVal_ &&
        in->getPrimitiveDesc() == inVal_->getPrimitiveDesc())
      << "primitive desc of input grad and value should be equal";

  // create reorder if has output grad does not match
  cpuInGrad_ = nullptr;
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
}

void MKLDNNConvLayer::resetWgtValBwdData(
    std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
    MKLDNNMatrixPtr& wgt) {
  if (dataPD == nullptr) {
    return;
  }

  // create new weight value for backward data, and create reorder if necessary
  // since the primitive_desc would be different with wgtVal_
  CHECK(wgtVal_) << "should have weight value";
  if (dataPD->weights_primitive_desc() != wgtVal_->getPrimitiveDesc()) {
    wgtValBwdData_ =
        MKLDNNMatrix::create(nullptr, dataPD->weights_primitive_desc());
    cvtWgtVal_ = MKLDNNMatrix::createReorder(wgtVal_, wgtValBwdData_);
    CHECK(cvtWgtVal_);
  } else {
    wgtValBwdData_ = wgtVal_;
  }
  VLOG(MKLDNN_FMTS) << "weight value format for backward data"
                    << wgtValBwdData_->getFormat();
}

}  // namespace paddle
