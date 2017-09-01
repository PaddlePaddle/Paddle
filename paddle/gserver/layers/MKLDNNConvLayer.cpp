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
#include "paddle/utils/Stat.h"

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
  auto targetDim = wgtVal_->getDims();
  // TODO: check g>2 cpu format
  CHECK_EQ(gp_, 1);
  auto srcFmt = (gp_ == 1) ? memory::format::oihw : memory::format::goihw;
  wgtVal_->reorderDataFrom(wgtVal_, srcFmt, targetDim);
  hasInitedWgt_ = true;
}

void MKLDNNConvLayer::convertWeightsToPaddle() {
  CHECK(wgtVal_) << "should have been initialized";
  LOG(FATAL) << "not implemented";
}

void MKLDNNConvLayer::reshape() {
  const Argument& input = getInput(0, getPrev(0)->getDeviceId());
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

  // cal output sizes
  // oc can not be changed
  oh_ = outputSize(ih_, fh_, ph_, sh_, caffeMode_);
  ow_ = outputSize(iw_, fw_, pw_, sw_, caffeMode_);
  printSizeInfo();

  // reset output
  output_.setFrameHeight(oh_);
  output_.setFrameWidth(ow_);
  resetOutput(bs_, oc_ * oh_ * ow_);

  // reset mkldnn forward
  resetFwd();
  needResetBwd_ = true;

  convertWeightsFromPaddle();
}

void MKLDNNConvLayer::resetFwd() {
  pipelineFwd_.clear();
  bool hasBias = biases_ && biases_->getW();

  // dims for conv
  memory::dims inDims = memory::dims{bs_, ic_, ih_, iw_};
  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  memory::dims wgtDims =
      (gp_ == 1) ? memory::dims{oc_, ic_, fh_, fw_}
                 : memory::dims{gp_, oc_ / gp_, ic_ / gp_, fh_, fw_};
  memory::dims biasDim = memory::dims{oc_};
  memory::dims strides = {sh_, sw_};
  memory::dims padding = {ph_, pw_};
  std::vector<int> padR = {ph_, pw_};
  for (int i = 0; i < 2; ++i) {
    if ((ih_ - ((fh_ - 1) * (dh_ + 1) + 1) + ph_ + padR[0]) / sh_ + 1 != oh_)
      ++padR[0];
    if ((iw_ - ((fw_ - 1) * (dw_ + 1) + 1) + pw_ + padR[1]) / sw_ + 1 != ow_)
      ++padR[1];
  }

  // create forward handle
  prop_kind pk = prop_kind::forward;
  algorithm algo = algorithm::convolution_direct;
  padding_kind padKind = padding_kind::zero;
  conv_fwd::desc fwdDesc =
      hasBias ? conv_fwd::desc(pk,
                               algo,
                               MKLDNNMatrix::createMemoryDesc(inDims),
                               MKLDNNMatrix::createMemoryDesc(wgtDims),
                               MKLDNNMatrix::createMemoryDesc(biasDim),
                               MKLDNNMatrix::createMemoryDesc(outDims),
                               strides,
                               padding,
                               padR,
                               padKind)
              : conv_fwd::desc(pk,
                               algo,
                               MKLDNNMatrix::createMemoryDesc(inDims),
                               MKLDNNMatrix::createMemoryDesc(wgtDims),
                               MKLDNNMatrix::createMemoryDesc(outDims),
                               strides,
                               padding,
                               padR,
                               padKind);
  conv_fwd::primitive_desc fwdPD = conv_fwd::primitive_desc(fwdDesc, engine_);

  // below are all internal memories
  const MatrixPtr& wgt = weight_->getW();
  const MatrixPtr& bias = hasBias ? biases_->getW() : nullptr;
  inVal_ = MKLDNNMatrix::create(nullptr, fwdPD.src_primitive_desc());
  outVal_ = MKLDNNMatrix::create(nullptr, fwdPD.dst_primitive_desc());
  wgtVal_ = MKLDNNMatrix::create(wgt, fwdPD.weights_primitive_desc());
  biasVal_ = hasBias ? MKLDNNMatrix::create(bias, fwdPD.bias_primitive_desc())
                     : nullptr;
  CHECK_EQ(fwdPD.bias_primitive_desc().desc().data.format, format::x)
      << "bias format should always be x";

  // add reorder if has user input value
  bool onlyMKLDNN = inputIsOnlyMKLDNN();
  int device = onlyMKLDNN ? MKLDNN_DEVICE : CPU_DEVICE;
  const MatrixPtr& in = getInputValue(0, device);
  if (onlyMKLDNN) {
    userInVal_ = std::dynamic_pointer_cast<MKLDNNMatrix>(in);
    CHECK(userInVal_) << "Input should be MKLDNNMatrix";
  } else {
    userInVal_ = MKLDNNMatrix::create(in, inDims, format::nchw, engine_);
  }
  cvtInVal_ = MKLDNNMatrix::createReorder(userInVal_, inVal_);
  if (cvtInVal_ == nullptr) {
    inVal_->updateData(userInVal_->getData());
  } else {
    pipelineFwd_.push_back(*cvtInVal_);
  }

  // add fwd handle
  if (hasBias) {
    fwd_.reset(new conv_fwd(fwdPD, *inVal_, *wgtVal_, *biasVal_, *outVal_));
  } else {
    fwd_.reset(new conv_fwd(fwdPD, *inVal_, *wgtVal_, *outVal_));
  }
  pipelineFwd_.push_back(*fwd_);

  // change original output value to mkldnn output value
  const MatrixPtr& out = output_.value;
  outVal_->updateData(out->getData());
  output_.value = std::dynamic_pointer_cast<Matrix>(outVal_);
  if (!outputIsOnlyMKLDNN()) {
    copyOutputInfoToOtherDevice();
    // find other cpu device and create reorder output to cpu device
    const MatrixPtr& cpuVal = getOutput(CPU_DEVICE).value;
    userOutVal_ = MKLDNNMatrix::create(cpuVal, outDims, format::nchw, engine_);
    cvtOutVal_ = MKLDNNMatrix::createReorder(outVal_, userOutVal_);
    if (cvtOutVal_ == nullptr) {
      userOutVal_->updateData(outVal_->getData());
    } else {
      pipelineFwd_.push_back(*cvtOutVal_);
    }
  }

  printValueFormatFlow();
}

void MKLDNNConvLayer::resetBwd() {
  LOG(FATAL) << "not implemented";
  if (!needResetBwd_) {
    return;
  }
  needResetBwd_ =
      false; /*
bool hasBias = biases_ && biases_->getWGrad();

/// backward weight
CHECK(inVal_) << "Should have input value";
const MatrixPtr& wgt = weight_->getWGrad();
const MatrixPtr& bias = hasBias ? biases_->getWGrad() : nullptr;

// TODO(TJ): merge outgrad
int device = outputIsOnlyMKLDNN() ? MKLDNN_DEVICE : CPU_DEVICE;
// for MKLDNN device:
// can not directly cast outputgrad to mkldnnmatrix,
// since each layer can not write the inputgrad to mkldnn inputgrad.
// So just create from matrix with outputvalue format.
// for CPU device:
// fc do not need to convert from cpu device since output is always nc format
// only need create from cpu device
const MatrixPtr& out = getOutput(device).grad;
outGrad_ = MKLDNNMatrix::create(out, outVal_->getPrimitiveDesc());
wgtGrad_ = MKLDNNMatrix::create(wgt, wgtVal_->getPrimitiveDesc());
biasGrad_ = hasBias ? MKLDNNMatrix::create(bias, biasVal_->getPrimitiveDesc())
           : nullptr;

// create memory primitive desc
fc_fwd::desc fwdDesc = fc_fwd::desc(prop_kind::forward,
                           inVal_->getMemoryDesc(),
                           wgtGrad_->getMemoryDesc(),
                           outGrad_->getMemoryDesc());
fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);
fc_bwdWgt::desc bwdWgtDesc = hasBias
                        ? fc_bwdWgt::desc(inVal_->getMemoryDesc(),
                                          wgtGrad_->getMemoryDesc(),
                                          biasGrad_->getMemoryDesc(),
                                          outGrad_->getMemoryDesc())
                        : fc_bwdWgt::desc(inVal_->getMemoryDesc(),
                                          wgtGrad_->getMemoryDesc(),
                                          outGrad_->getMemoryDesc());
fc_bwdWgt::primitive_desc bwdWgtPD =
fc_bwdWgt::primitive_desc(bwdWgtDesc, engine_, fwdPD);

if (hasBias) {
bwdWgt_.reset(
new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_, *biasGrad_));
} else {
bwdWgt_.reset(new fc_bwdWgt(bwdWgtPD, *inVal_, *outGrad_, *wgtGrad_));
}
pipelineBwd_.clear();
pipelineBwd_.push_back(*bwdWgt_);

/// backward data
device = inputIsOnlyMKLDNN() ? MKLDNN_DEVICE : CPU_DEVICE;
const MatrixPtr& in = getInputGrad(0, device);
if (in == nullptr) {
return;
}
if (getInput(0, device).getAllCount() > 1) {
// TODO(TJ): use outputMaps_ ways when merge outgrad done
} else {
inGrad_ = MKLDNNMatrix::create(in, inVal_->getPrimitiveDesc());
}

fc_bwdData::desc bwdDataDesc = fc_bwdData::desc(inVal_->getMemoryDesc(),
                                       wgtGrad_->getMemoryDesc(),
                                       outGrad_->getMemoryDesc());
fc_bwdData::primitive_desc bwdDataPD =
fc_bwdData::primitive_desc(bwdDataDesc, engine_, fwdPD);

CHECK(wgtVal_) << "Should have weight memory";
bwdData_.reset(new fc_bwdData(bwdDataPD, *outGrad_, *wgtVal_, *inGrad_));
printGradFormatFlow();
pipelineBwd_.push_back(*bwdData_);*/
}

void MKLDNNConvLayer::forward(PassType passType) {
  Layer::forward(passType);
  reshape();

  {
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
    syncInputValue();

    // just submit forward pipeline
    stream_->submit(pipelineFwd_);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwActTimer", getName().c_str());
    forwardActivation();
  }
}

void MKLDNNConvLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpActTimer", getName().c_str());
    backwardActivation();
  }

  {
    REGISTER_TIMER_INFO("mkldnn_bwdTimer", getName().c_str());
    resetBwd();

    syncOutputGrad();
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
