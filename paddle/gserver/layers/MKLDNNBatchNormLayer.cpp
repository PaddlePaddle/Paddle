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

#include "MKLDNNBatchNormLayer.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_batch_norm, MKLDNNBatchNormLayer);

bool MKLDNNBatchNormLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  // first one is input layer
  // the other two are created in config_parser.py saving moving mean and var
  CHECK_EQ(inputLayers_.size(), 3U);
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK_EQ(inputLayers_.size(), size_t(config_.inputs_size()));

  const ImageConfig& conf = config_.inputs(0).image_conf();
  ic_ = conf.channels();
  ih_ = inputLayers_[0]->getOutput().getFrameHeight();
  iw_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (iw_ == 0 && ih_ == 0) {
    iw_ = conf.img_size();
    ih_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  }
  oc_ = ic_;
  oh_ = ih_;
  ow_ = iw_;
  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  movingAvgFraction_ = config_.moving_average_fraction();
  epsilon_ = config_.epsilon();

  VLOG(MKLDNN_BASE) << "--- " << (useGlobalStats_ ? "use" : "do not use")
                    << " --- global stats";
  VLOG(MKLDNN_BASE) << "Moving average fraction: " << movingAvgFraction_;

  initWeight();
  movingMean_.reset(new Weight(oc_, 1, parameters_[1], 0));
  movingVar_.reset(new Weight(oc_, 1, parameters_[2], 0));
  return true;
}

void MKLDNNBatchNormLayer::initWeight() {
  weight_.reset(new Weight(1, oc_, parameters_[0]));
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
  }
  CHECK_EQ(weight_ != nullptr, biases_ != nullptr)
      << "only support have both weight and bias, or neither";
  if (weight_ && weight_->getW()) {
    CHECK(biases_ && biases_->getW());
    valueScaleShift_ = Matrix::create(2, oc_, false, false);
    valueScaleShift_->zeroMem();
    VectorPtr scale(new CpuVector(oc_, valueScaleShift_->getMemoryHandle(), 0));
    VectorPtr shift(
        new CpuVector(oc_, valueScaleShift_->getMemoryHandle(), oc_));
    const VectorPtr& wgt = parameters_[0]->getBuf(PARAMETER_VALUE);
    const VectorPtr& bias = biasParameter_->getBuf(PARAMETER_VALUE);
    scale->copyFrom(*wgt);
    shift->copyFrom(*bias);
    wgt->setData(valueScaleShift_->getData());
    bias->setData(valueScaleShift_->getData() + oc_);
  }
  if (weight_ && weight_->getWGrad()) {
    CHECK(biases_ && biases_->getWGrad());
    gradScaleShift_ = Matrix::create(2, oc_, false, false);
    gradScaleShift_->zeroMem();
    const VectorPtr& wgt = parameters_[0]->getBuf(PARAMETER_GRADIENT);
    const VectorPtr& bias = biasParameter_->getBuf(PARAMETER_GRADIENT);
    wgt->setData(gradScaleShift_->getData());
    bias->setData(gradScaleShift_->getData() + oc_);
  }
}

void MKLDNNBatchNormLayer::convertWeightsFromPaddle() {
  if (hasInitedWgt_) {
    return;
  }
  // prepare mean and var if necessary
  if (useGlobalStats_) {
    CHECK(mean_);
    CHECK(var_);
    mean_->copyFrom(*(movingMean_->getW()));
    var_->copyFrom(*(movingVar_->getW()));
  }
  hasInitedWgt_ = true;
}

void MKLDNNBatchNormLayer::calMovingMeanAndVar() {
  // calculating and saving moving mean and variance
  CHECK_EQ(useGlobalStats_, false);
  movingMean_->getW()->add(
      *mean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  // here var is v^2
  movingVar_->getW()->add(*var_, movingAvgFraction_, 1.0 - movingAvgFraction_);
}

void MKLDNNBatchNormLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);
  oh = ih;
  ow = iw;
  // ic_ and oc can not be changed
  CHECK_EQ((size_t)ic,
           inputLayers_[0]->getOutputValue()->getElementCnt() / bs / ih / iw)
      << "Input channel can not be changed";
  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNBatchNormLayer::resetFwd(std::vector<primitive>& pipeline,
                                    std::vector<MKLDNNMatrixPtr>& inputs,
                                    MKLDNNMatrixPtr& out) {
  // In training phase, it will always calculate mean and var,
  // so useGlobalStats must be false.
  // In scoring phase, it depends on useGlobalStats choice.
  if (passType_ != PASS_TEST && useGlobalStats_ == true) {
    LOG(WARNING) << "use_global_stats is invalid setting in training phase";
    useGlobalStats_ = false;
  }

  resetFwdBuffers(inputs[0], wgtVal_, out);

  resetFwdPD(fwdPD_, inputs[0], wgtVal_, out);

  resetFwdPipeline(pipeline, fwdPD_, inputs[0], wgtVal_, out);
}

void MKLDNNBatchNormLayer::resetBwd(std::vector<primitive>& pipeline,
                                    std::vector<MKLDNNMatrixPtr>& inputs,
                                    MKLDNNMatrixPtr& out) {
  std::shared_ptr<bn_bwd::primitive_desc> pd;

  resetBwdBuffers(inputs[0], wgtGrad_, out);

  resetBwdPD(pd, inputs[0], wgtGrad_, out);

  resetBwdPipeline(pipeline, pd, inputs[0], wgtGrad_, out);
}

void MKLDNNBatchNormLayer::forward(PassType passType) {
  MKLDNNLayer::forward(passType);

  // calculate and save moving mean and variance
  if (passType_ != PASS_TEST) {
    calMovingMeanAndVar();
  }
}

void MKLDNNBatchNormLayer::updateWeights(const UpdateCallback& callback) {
  weight_->getParameterPtr()->incUpdate(callback);
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

void MKLDNNBatchNormLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                           MKLDNNMatrixPtr& wgt,
                                           MKLDNNMatrixPtr& out) {
  resetInValue(in);

  memory::dims outDims = memory::dims{bs_, oc_, oh_, ow_};
  CHECK(in);
  auto outPD =
      MKLDNNMatrix::createPrimitiveDesc(outDims, in->getFormat(), engine_);
  resetOutValue(out, outPD);

  if (valueScaleShift_) {
    auto pd = MKLDNNMatrix::createPrimitiveDesc({2, oc_}, format::nc, engine_);
    resetWithMatrix(wgt, valueScaleShift_, pd);
  }
  if (passType_ != PASS_TEST || useGlobalStats_) {
    auto pd = MKLDNNMatrix::createPrimitiveDesc({oc_}, format::x, engine_);
    mean_ = MKLDNNMatrix::create(pd);
    var_ = MKLDNNMatrix::create(pd);
  }
}

void MKLDNNBatchNormLayer::resetFwdPD(
    std::shared_ptr<bn_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr in,
    MKLDNNMatrixPtr wgt,
    MKLDNNMatrixPtr out) {
  flags_ = 0u;
  prop_kind pk = passType_ == PASS_TEST ? prop_kind::forward_scoring
                                        : prop_kind::forward_training;
  if (useGlobalStats_) {
    flags_ = (flags_ | batch_normalization_flag::use_global_stats);
  }
  if (wgt) {
    flags_ = (flags_ | batch_normalization_flag::use_scale_shift);
  }
  auto fwdDesc = bn_fwd::desc(pk, in->getMemoryDesc(), epsilon_, flags_);
  pd.reset(new bn_fwd::primitive_desc(fwdDesc, engine_));
  CHECK_PRIMITIVE_DESC_EQ(out, pd->dst_primitive_desc());
  if (wgt) {
    CHECK_PRIMITIVE_DESC_EQ(wgt, pd->weights_primitive_desc());
  }
  if (passType_ != PASS_TEST || useGlobalStats_) {
    CHECK_PRIMITIVE_DESC_EQ(mean_, pd->mean_primitive_desc());
    CHECK_PRIMITIVE_DESC_EQ(var_, pd->variance_primitive_desc());
  }
}

void MKLDNNBatchNormLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<bn_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& out) {
  if (passType_ == PASS_TEST) {
    if (useGlobalStats_) {
      fwd_.reset(wgt != nullptr ? new bn_fwd(*pd,
                                             *in,
                                             (const primitive::at)(*mean_),
                                             (const primitive::at)(*var_),
                                             *wgt,
                                             *out)
                                : new bn_fwd(*pd,
                                             *in,
                                             (const primitive::at)(*mean_),
                                             (const primitive::at)(*var_),
                                             *out));
    } else {
      fwd_.reset(wgt != nullptr ? new bn_fwd(*pd, *in, *wgt, *out)
                                : new bn_fwd(*pd, *in, *out));
    }
  } else {
    CHECK_EQ(useGlobalStats_, false)
        << "useGlobalStats should be false in training";
    fwd_.reset(wgt != nullptr ? new bn_fwd(*pd, *in, *wgt, *out, *mean_, *var_)
                              : new bn_fwd(*pd, *in, *out, *mean_, *var_));
  }
  pipeline.push_back(*fwd_);
}

void MKLDNNBatchNormLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                           MKLDNNMatrixPtr& wgt,
                                           MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0] && outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  resetInGrad(in, inVals_[0]->getPrimitiveDesc());
  if (gradScaleShift_) {
    CHECK(wgtVal_);
    resetWithMatrix(wgt, gradScaleShift_, wgtVal_->getPrimitiveDesc());
  }
}

void MKLDNNBatchNormLayer::resetBwdPD(
    std::shared_ptr<bn_bwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& out) {
  pd = nullptr;
  if (in == nullptr) {
    return;
  }
  CHECK_PRIMITIVE_DESC_EQ(out, in->getPrimitiveDesc());
  auto md = in->getMemoryDesc();
  auto bwdDesc = bn_bwd::desc(prop_kind::backward, md, md, epsilon_, flags_);
  pd.reset(new bn_bwd::primitive_desc(bwdDesc, engine_, *fwdPD_));
  CHECK(pd->weights_primitive_desc() == fwdPD_->weights_primitive_desc());
  CHECK_PRIMITIVE_DESC_EQ(wgt, pd->diff_weights_primitive_desc());
  CHECK_PRIMITIVE_DESC_EQ(mean_, pd->mean_primitive_desc());
  CHECK_PRIMITIVE_DESC_EQ(var_, pd->variance_primitive_desc());
}

void MKLDNNBatchNormLayer::resetBwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<bn_bwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& wgt,
    MKLDNNMatrixPtr& out) {
  if (pd == nullptr) {
    return;
  }
  CHECK(inVals_[0]);
  bwdData_.reset(
      wgt && wgtVal_
          ? new bn_bwd(
                *pd, *inVals_[0], *mean_, *var_, *out, *wgtVal_, *in, *wgt)
          : new bn_bwd(*pd, *inVals_[0], *mean_, *var_, *out, *in));
  pipeline.push_back(*bwdData_);
}

}  // namespace paddle
