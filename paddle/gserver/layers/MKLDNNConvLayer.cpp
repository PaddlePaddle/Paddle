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
  CHECK_EQ(inputLayers_.size(), 1UL) << "Only support one input layer yet";
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
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_, 0));
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
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);

  // cal output sizes
  // oc can not be changed
  int fh = (fh_ - 1) * dh_ + 1;
  int fw = (fw_ - 1) * dw_ + 1;
  oh = outputSize(ih, fh, ph_, sh_, caffeMode_);
  ow = outputSize(iw, fw, pw_, sw_, caffeMode_);

  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNConvLayer::resetFwd(std::vector<primitive>& pipeline,
                               std::vector<MKLDNNMatrixPtr>& inputs,
                               MKLDNNMatrixPtr& out) {
  resetFwdPD(fwdPD_);

  resetFwdBuffers(fwdPD_, inputs[0], wgtVal_, biasVal_, out);

  resetFwdPipeline(pipeline, fwdPD_, inputs[0], wgtVal_, biasVal_, out);
}

void MKLDNNConvLayer::resetBwd(std::vector<primitive>& pipeline,
                               std::vector<MKLDNNMatrixPtr>& inputs,
                               MKLDNNMatrixPtr& out) {
  std::shared_ptr<conv_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<conv_bwdData::primitive_desc> bwdDataPD;

  resetBwdWgtPD(bwdWgtPD);

  resetBwdDataPD(bwdDataPD);

  resetBwdBuffers(bwdWgtPD, bwdDataPD, inputs[0], wgtGrad_, biasGrad_, out);

  resetBwdPipeline(
      pipeline, bwdWgtPD, bwdDataPD, inputs[0], wgtGrad_, biasGrad_, out);
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
  resetInValue(
      in, std::make_shared<memory::primitive_desc>(pd->src_primitive_desc()));

  resetOutValue(out, pd->dst_primitive_desc());

  resetWithMatrix(wgt, weight_->getW(), pd->weights_primitive_desc());

  if (biases_ && biases_->getW()) {
    resetWithMatrix(bias, biases_->getW(), pd->bias_primitive_desc());
  } else {
    bias = nullptr;
  }
}

void MKLDNNConvLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<conv_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  if (bias) {
    fwd_.reset(new conv_fwd(*pd, *in, *wgt, *bias, *out));
  } else {
    fwd_.reset(new conv_fwd(*pd, *in, *wgt, *out));
  }
  pipeline.push_back(*fwd_);
}

void MKLDNNConvLayer::resetBwdWgtPD(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& pd) {
  memory::dims wgtDims, biasDims, strides, dilations, padL, padR;
  loadConvSettings(wgtDims, biasDims, strides, dilations, padL, padR);

  // create backward weight using input, output and weight value memory desc
  CHECK(inVals_[0]) << "Should have internal input value";
  CHECK(outVal_) << "Should have internal output value";
  CHECK(wgtVal_) << "Should have weight value";
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  auto bwdWgtDesc = biasVal_ != nullptr
                        ? conv_bwdWgt::desc(algo,
                                            inVals_[0]->getMemoryDesc(),
                                            wgtVal_->getMemoryDesc(),
                                            biasVal_->getMemoryDesc(),
                                            outVal_->getMemoryDesc(),
                                            strides,
                                            padL,
                                            padR,
                                            padKind)
                        : conv_bwdWgt::desc(algo,
                                            inVals_[0]->getMemoryDesc(),
                                            wgtVal_->getMemoryDesc(),
                                            outVal_->getMemoryDesc(),
                                            strides,
                                            padL,
                                            padR,
                                            padKind);
  pd.reset(new conv_bwdWgt::primitive_desc(bwdWgtDesc, engine_, *fwdPD_));
  CHECK_PRIMITIVE_DESC_EQ(inVals_[0], pd->src_primitive_desc());
  CHECK_PRIMITIVE_DESC_EQ(
      outVal_,
      pd->diff_dst_primitive_desc(),
      "primitive desc of out value and grad should be equal");
  CHECK_PRIMITIVE_DESC_EQ(
      wgtVal_,
      pd->diff_weights_primitive_desc(),
      "primitive desc of weight value and grad should be equal");
}

void MKLDNNConvLayer::resetBwdDataPD(
    std::shared_ptr<conv_bwdData::primitive_desc>& pd) {
  pd = nullptr;
  if (inputLayers_[0]->getOutput().grad == nullptr) {
    return;
  }

  memory::dims wgtDims, biasDims, strides, dilations, padL, padR;
  loadConvSettings(wgtDims, biasDims, strides, dilations, padL, padR);
  CHECK(inVals_[0]) << "Should have internal input value";
  CHECK(outVal_) << "Should have internal output value";
  // create backward data using input and output value memory desc
  // but using weight memory desc with any format
  auto bwdDataDesc = conv_bwdData::desc(algorithm::convolution_direct,
                                        inVals_[0]->getMemoryDesc(),
                                        MKLDNNMatrix::createMemoryDesc(wgtDims),
                                        outVal_->getMemoryDesc(),
                                        strides,
                                        padL,
                                        padR,
                                        padding_kind::zero);
  pd.reset(new conv_bwdData::primitive_desc(bwdDataDesc, engine_, *fwdPD_));
  CHECK_PRIMITIVE_DESC_EQ(
      inVals_[0],
      pd->diff_src_primitive_desc(),
      "primitive desc of in value and grad should be equal");
  CHECK_PRIMITIVE_DESC_EQ(
      outVal_,
      pd->diff_dst_primitive_desc(),
      "primitive desc of out value and grad should be equal");
}

void MKLDNNConvLayer::resetBwdBuffers(
    std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
    std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  CHECK(wgtPD);
  resetOutGrad(out, wgtPD->diff_dst_primitive_desc());

  resetWithMatrix(
      wgt, weight_->getWGrad(), wgtPD->diff_weights_primitive_desc());
  CHECK_PRIMITIVE_DESC_EQ(
      wgtVal_,
      wgt->getPrimitiveDesc(),
      "primitive desc of weight grad and value should be equal");

  bias = nullptr;
  if (biases_ && biases_->getWGrad()) {
    resetWithMatrix(
        bias, biases_->getWGrad(), wgtPD->diff_bias_primitive_desc());
    CHECK(bias);
    CHECK_PRIMITIVE_DESC_EQ(
        biasVal_,
        bias->getPrimitiveDesc(),
        "primitive desc of bias grad and value should be equal");
  }

  if (dataPD == nullptr) {
    return;
  }
  resetInGrad(in, dataPD->diff_src_primitive_desc());
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
  CHECK(inVals_[0]);
  // add bwdWgt handle
  if (bias) {
    bwdWgt_.reset(new conv_bwdWgt(*wgtPD, *inVals_[0], *out, *wgt, *bias));
  } else {
    bwdWgt_.reset(new conv_bwdWgt(*wgtPD, *inVals_[0], *out, *wgt));
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
    wgtValBwdData_ = MKLDNNMatrix::create(dataPD->weights_primitive_desc());
    cvtWgtVal_ = MKLDNNMatrix::createReorder(wgtVal_, wgtValBwdData_);
    CHECK(cvtWgtVal_);
  } else {
    wgtValBwdData_ = wgtVal_;
  }
  VLOG(MKLDNN_FMTS) << "weight value format for backward data: "
                    << wgtValBwdData_->getFormat();
}

}  // namespace paddle
