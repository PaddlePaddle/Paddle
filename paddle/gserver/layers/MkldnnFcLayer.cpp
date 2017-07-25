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

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnFcLayer.h"

using namespace mkldnn;  // NOLINT
typedef mkldnn::inner_product_forward fc_fwd;
typedef mkldnn::inner_product_backward_weights fc_bwdWgt;
typedef mkldnn::inner_product_backward_data fc_bwdData;

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

// load the settings from proto
void MkldnnFcLayer::loadConfig() {
  MkldnnLayer::loadConfig();

  // get dim of input and output
  const FCConfig &conf = config_.inputs(0).fc_conf();
  dim_in_ = conf.dim_in();
  dim_out_ = conf.dim_out();
}

bool MkldnnFcLayer::initWgt(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  CHECK_EQ(inputLayers_.size(), 1) << "Only support one input layer yet!";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK(!parameters_[0]->isSparse()) << "Do not support sparse yet";
  CHECK_EQ(dim_out_, getSize());
  CHECK_EQ(parameters_[0]->getSize(), dim_out_ * dim_in_);

  // create weight
  weight_ = std::unique_ptr<Weight>(
              new Weight(dim_out_, dim_in_, parameters_[0], 0));

  // The weight_ is transposed from initial paddle weight
  paddleWgt_ = Matrix::create(weight_->getW()->getData(),
    dim_in_, dim_out_, false, false);

  // create biases
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, dim_out_, biasParameter_));
    hasBias_ = true;
  }

  return true;
}

void MkldnnFcLayer::reshape() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  CHECK_EQ(dim_in_, iMatW_) << "should not change input layer size,"
    << "this would need to change the weight size which is fixed";

  // FC layer do not care about the seqlen changing
  // the bs would not be actually used,
  // use oMatH_ instead as the bs in MKLDNN
  reshapeOutMatSize();

  reshapeBatchSize();

  reshapeImgSize();

  setOutImgSize();
}

void MkldnnFcLayer::resetFwd(PassType passType) {
  CHECK_EQ(inputLayers_.size(), 1);
  CHECK_EQ(hasBias_, biases_ && biases_->getW());
  engine_ = CpuEngine::Instance().getEngine();
  std::shared_ptr<fc_fwd::primitive_desc> fwdPD;

  resetDnnBufferShapes();

  resetDnnFwdPD(fwdPD);

  resetDnnFwdBuffers(fwdPD);

  resetFwdPipeline(fwdPD);

  initWgtFromPaddle();
}

void MkldnnFcLayer::submitFwd() {
  forwardDnnVal();

  forwardDnnAct();
}

void MkldnnFcLayer::resetBwd() {
  CHECK_EQ(hasBias_, biases_ && biases_->getWGrad());
  std::shared_ptr<fc_bwdWgt::primitive_desc> bwdWgtPD;
  std::shared_ptr<fc_bwdData::primitive_desc> bwdDataPD;

  resetDnnBwdWgtPD(bwdWgtPD);

  resetDnnBwdDataPD(bwdDataPD);

  resetDnnBwdBuffers(bwdWgtPD, bwdDataPD);

  resetDnnBwdPipeline(bwdWgtPD, bwdDataPD);
}

void MkldnnFcLayer::submitBwd(const UpdateCallback &callback) {
  BackwardDnnAct();

  backwardDnnVal();

  updateParameter(callback);
}

/*************************** protected methods: *******************************/
/// reshape the output matrix size according to input matrix size
void MkldnnFcLayer::reshapeOutMatSize() {
  // keep the output dim unchanged
  oMatW_ = dim_out_;
  CHECK_EQ(oMatW_, getSize());
  // config_.set_size(oMatW_);

  // only can change matrix height
  oMatH_ = iMatH_;
}

void MkldnnFcLayer::reshapeBatchSize() {
  int seqLen = getInput(0).getMklSeqLen();
  if (seqLen > 1) {
    bs_ = oMatH_ / seqLen;
    CHECK_EQ(bs_ * seqLen, oMatH_) << "not divisible";
  } else {
    bs_ = oMatH_;
  }
}

void MkldnnFcLayer::reshapeImgSize() {
  // reshape input sizes
  const Argument& input = getInput(0);
  ih_ = input.getFrameHeight();
  iw_ = input.getFrameWidth();
  if (ih_ == 0) {
    ih_ = 1;
  }
  if (iw_ == 0) {
    iw_ = 1;
  }
  hasSpatial_ = true;
  if (ih_ == 1 && iw_ == 1) {
    hasSpatial_ = false;
  }
  ic_ = iMatW_ / (ih_ * iw_);
  CHECK_EQ(ic_ * ih_ * iw_, iMatW_) << "not divisible";

  oc_ = oMatW_;
  CHECK_EQ(oc_, dim_out_) << "output channel can not be changed";
  oh_ = 1;
  ow_ = 1;
}

void MkldnnFcLayer::setOutImgSize() {
  CHECK_EQ(oc_, oMatW_) << "output layersize can not be changed";
  CHECK_EQ(oh_, 1);
  CHECK_EQ(ow_, 1);

  output_.setFrameHeight(oh_);
  output_.setFrameWidth(ow_);
}

void MkldnnFcLayer::resetDnnBufferShapes() {
  if (hasSpatial_) {
    // use output matrix height instead of batchsize
    botDims_ = {(int)(oMatH_), ic_, ih_, iw_};
    botFmt_ = memory::format::nchw;
    wgtDims_ = {oc_, ic_, ih_, iw_};
    wgtFmt_ = memory::format::oihw;
  } else {
    botDims_ = {(int)oMatH_, ic_};
    botFmt_ = memory::format::nc;
    wgtDims_ = {oc_, ic_};
    wgtFmt_ = memory::format::oi;
  }

  topDims_ = {(int)oMatH_, oc_};
  topFmt_ = memory::format::nc;

  if (hasBias_) {
    biasDims_ = {oc_};
    biasFmt_ = memory::format::x;
  } else {
    biasDims_ = {};
    biasFmt_ = memory::format::format_undef;
  }
}

void MkldnnFcLayer::resetDnnFwdPD(
  std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  prop_kind pk = prop_kind::forward;
  std::shared_ptr<fc_fwd::desc> fwdDesc;
  if (hasBias_) {
    fwdDesc.reset(new fc_fwd::desc(pk,
        MkldnnBuffer::getMD(botDims_),
        MkldnnBuffer::getMD(wgtDims_),
        MkldnnBuffer::getMD(biasDims_),
        MkldnnBuffer::getMD(topDims_)));
  } else {
    fwdDesc.reset(new fc_fwd::desc(pk,
        MkldnnBuffer::getMD(botDims_),
        MkldnnBuffer::getMD(wgtDims_),
        MkldnnBuffer::getMD(topDims_)));
  }
  fwdPD.reset(new fc_fwd::primitive_desc(*fwdDesc, engine_));
}

void MkldnnFcLayer::resetDnnFwdBuffers(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  CHECK(fwdPD);
  CHECK(getInput(0).value) << "The input of mkldnn fc layer must be matrix";

  resetDnnBotData(fwdPD);

  resetDnnTopData(fwdPD);

  resetDnnWgtBiasData(fwdPD);
}

void MkldnnFcLayer::resetDnnBotData(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  botData_.reset(new MkldnnBuffer());
  const MatrixPtr& botVal = getInputValue(0);
  real *botValData = botVal->getData();
  if (prevIsDnn_[0]) {
    const MkldnnBufferPtr prvTop =
        std::static_pointer_cast<MkldnnBuffer> (getPrev(0)->getMkldnnTopData());
    CHECK(prvTop) << "prev layer should have dnn buffer.";
    botData_->resetUser(prvTop->getUser());
    VLOG(DNN_FMTS) << "use prev data fmt: " << botData_->getUserFmt();
  } else {
    botData_->resetUser(botValData, botDims_, botFmt_, engine_);
  }
  botData_->resetIntl(fwdPD->src_primitive_desc());
  botData_->resetReorder(dnnUser2Intl);
}

void MkldnnFcLayer::resetDnnTopData(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  topData_.reset(new MkldnnBuffer());
  const MatrixPtr& topVal = getOutputValue();
  real *topValData = topVal->getData();
  if (nextIsDnn_) {
    topData_->resetUser(topValData, fwdPD->dst_primitive_desc());
    topData_->resetIntl(topData_->getUser());
  } else {
    topData_->resetUser(topValData, topDims_, topFmt_, engine_);
    topData_->resetIntl(fwdPD->dst_primitive_desc());
    topData_->resetReorder(dnnIntl2User);
  }
}

void MkldnnFcLayer::resetDnnWgtBiasData(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  // weight
  wgtData_.reset(new MkldnnBuffer());
  const MatrixPtr& wgtVal = weight_->getW();
  real *wgtValData = wgtVal->getData();
  wgtData_->resetUser(wgtValData, fwdPD->weights_primitive_desc());
  wgtData_->resetIntl(wgtData_->getUser());
  CHECK_EQ(wgtData_->getIntlSize(), parameters_[0]->getSize())
    << "can not use mkldnn wgt since memory size does not equal";
  VLOG(DNN_FMTS) << "weight format: " << wgtData_->getIntlFmt();

  // bias
  if (!hasBias_) {
    return;
  }
  biasData_.reset(new MkldnnBuffer());
  real *biasValData = biases_->getW()->getData();
  biasData_->resetUser(biasValData, biasDims_, biasFmt_, engine_);
  CHECK(biasData_->getUserPD() == fwdPD->bias_primitive_desc())
    << "should always be format::x, or changed in later mkldnn version";
  biasData_->resetIntl(biasData_->getUser());
}

void MkldnnFcLayer::resetFwdPipeline(
  const std::shared_ptr<fc_fwd::primitive_desc>& fwdPD) {
  if (hasBias_) {
    fwd_.reset(new fc_fwd(*fwdPD,
      *(botData_->getIntl()), *(wgtData_->getIntl()),
      *(biasData_->getIntl()), *(topData_->getIntl())));
  } else {
    fwd_.reset(new fc_fwd(*fwdPD,
      *(botData_->getIntl()), *(wgtData_->getIntl()),
      *(topData_->getIntl())));
  }

  if (botData_->needReorder()) {
    pipelineFwd_.push_back(*botData_->getReorder());
  }
  CHECK_EQ(wgtData_->needReorder(), false) << "wgt should not need reorder!";
  pipelineFwd_.push_back(*fwd_);
  if (topData_->needReorder()) {
    pipelineFwd_.push_back(*topData_->getReorder());
  }
}

void MkldnnFcLayer::initWgtFromPaddle() {
  if (passType_ == PASS_TEST && !scoreWithPaddleWgt_) {
    return;
  }
  
  if (hasInitedWgt_) {
    return;
  }
  
  // Firstly in mkldnn, the matrix is transposed from initial paddle weight
  MatrixPtr paddleWgtT;
  paddleWgt_->transpose(paddleWgtT, true);

  // Then, reorder the format from transpoesd matrix to mkldnn intl memory
  MkldnnBufferPtr cvtWgt(new MkldnnBuffer());
  cvtWgt->resetUser(paddleWgtT->getData(), wgtDims_, wgtFmt_, engine_);
  cvtWgt->resetIntl(wgtData_->getIntl());
  cvtWgt->resetReorder(dnnUser2Intl);

  // start cvt
  std::vector<primitive> cvtToDnnWgt;
  CHECK(cvtWgt->needReorder()) << "should always need cvt from paddle weight";
  cvtToDnnWgt.push_back(*cvtWgt->getReorder());
  stream(stream::kind::eager).submit(cvtToDnnWgt).wait();
  
  hasInitedWgt_ = true;
}

void MkldnnFcLayer::forwardDnnVal() {
  real *botValData = getPrev(0)->getOutputValue()->getData();
  botData_->updateUserData(botValData);
  stream(stream::kind::eager).submit(pipelineFwd_).wait();
}

/*************************** for backward methods: ****************************/
void MkldnnFcLayer::resetDnnBwdWgtPD(
  std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD) {
  std::shared_ptr<fc_fwd::primitive_desc> bwdFwdPD;
  std::shared_ptr<fc_bwdWgt::desc> bwdWgtDesc;

  getBwdFwdPD(bwdFwdPD);

  if (hasBias_) {
    bwdWgtDesc.reset(new fc_bwdWgt::desc(
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      biasData_->getIntlMD(),
      MkldnnBuffer::getMD(topDims_)));
  } else {
    bwdWgtDesc.reset(new fc_bwdWgt::desc(
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
  }
  bwdWgtPD.reset(new fc_bwdWgt::primitive_desc(
    *bwdWgtDesc, engine_, *bwdFwdPD));
  CHECK(botData_->getIntlPD() == bwdWgtPD->src_primitive_desc());
  if (hasBias_) {
    CHECK(biasData_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
      << "should always be format::x, or changed in later mkldnn version";
  }
}

void MkldnnFcLayer::resetDnnBwdDataPD(
  std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  if (!hasBotGrad()) {
    return;
  }

  std::shared_ptr<fc_fwd::primitive_desc> bwdFwdPD;
  std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;

  getBwdFwdPD(bwdFwdPD);

  bwdDataDesc.reset(new inner_product_backward_data::desc(
    MkldnnBuffer::getMD(botDims_),
    wgtData_->getIntlMD(),
    MkldnnBuffer::getMD(topDims_)));
  bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
    *bwdDataDesc, engine_, *bwdFwdPD));

// CHECK(botData_->getIntlPD() == bwdDataPD->diff_src_primitive_desc());
  CHECK(wgtData_->getIntlPD() == bwdDataPD->weights_primitive_desc());
// CHECK(topDiffBwdWgt_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
}

void MkldnnFcLayer::getBwdFwdPD(
  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& bwdFwdPD) {
  prop_kind pk = prop_kind::forward;
  std::shared_ptr<fc_fwd::desc> bwdFwdDesc;
  bwdFwdDesc.reset(new fc_fwd::desc(pk,
      botData_->getIntlMD(),
      MkldnnBuffer::getMD(wgtDims_),
      MkldnnBuffer::getMD(topDims_)));
  bwdFwdPD.reset(new fc_fwd::primitive_desc(*bwdFwdDesc, engine_));
}

void MkldnnFcLayer::resetDnnBwdBuffers(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  // topdiff buffer in bwdwgt may have differen format with bwddata
  // so have two different buffer
  resetDnnTopDiffBwdData(bwdDataPD);

  resetDnnTopDiffBwdWgt(bwdWgtPD);

  resetDnnWgtBiasDiff(bwdWgtPD);

  resetDnnBotDiff(bwdDataPD);
}

void MkldnnFcLayer::resetDnnTopDiffBwdData(
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  if (!hasBotGrad()) {
    return;
  }

  CHECK(bwdDataPD);
  topDiff_.reset(new MkldnnBuffer());
  real *topGradData = getOutputGrad()->getData();
  if (nextIsDnn_) {
    const MkldnnBufferPtr nextBotDiff = std::static_pointer_cast<MkldnnBuffer>
      (nextLayers_[0]->getMkldnnBotDiff());
    CHECK(nextBotDiff) << "next layer should have dnn buffer.";
    topDiff_->resetUser(nextBotDiff->getUser());
    VLOG(DNN_FMTS) << "topdiff use next diff fmt: " << topDiff_->getUserFmt();
  } else {
    topDiff_->resetUser(topGradData, topDims_, topFmt_, engine_);
  }
  topDiff_->resetIntl(bwdDataPD->diff_dst_primitive_desc());
  topDiff_->resetReorder(dnnUser2Intl);
}

void MkldnnFcLayer::resetDnnTopDiffBwdWgt(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD) {
  CHECK(bwdWgtPD);
  topDiffBwdWgt_.reset(new MkldnnBuffer());
  real *topGradData = getOutputGrad()->getData();
  if (nextIsDnn_) {
    const MkldnnBufferPtr nextBotDiff = std::static_pointer_cast<MkldnnBuffer>
      (nextLayers_[0]->getMkldnnBotDiff());
    CHECK(nextBotDiff) << "next layer should have dnn buffer.";
    topDiffBwdWgt_->resetUser(nextBotDiff->getUser());
    VLOG(DNN_FMTS) << "topdiffBwdWgt use next diff fmt: "
      << topDiffBwdWgt_->getUserFmt();
  } else {
    topDiffBwdWgt_->resetUser(topGradData, topDims_, topFmt_, engine_);
  }
  topDiffBwdWgt_->resetIntl(bwdWgtPD->diff_dst_primitive_desc());
  topDiffBwdWgt_->resetReorder(dnnUser2Intl);
  // topdiff for bwdwgt may differ for bwddata
  VLOG(DNN_FMTS) << "topdiff for bwd weight flow --- "
    << topDiffBwdWgt_->getIntlFmt()
    << " <<< "
    << topDiffBwdWgt_->getUserFmt();
}

void MkldnnFcLayer::resetDnnWgtBiasDiff(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD) {
  CHECK(bwdWgtPD);
  CHECK(weight_->getWGrad()) << "should have weight grad anyway";
  wgtDiff_.reset(new MkldnnBuffer());
  real *wgtGradData = weight_->getWGrad()->getData();
  wgtDiff_->resetUser(wgtGradData, bwdWgtPD->diff_weights_primitive_desc());
  wgtDiff_->resetIntl(wgtDiff_->getUser());
  CHECK_EQ(wgtDiff_->getIntlSize(), wgtData_->getIntlSize())
    << "can not use mkldnn wgt since memory size does not equal";
  CHECK(wgtDiff_->getUserPD() == wgtDiff_->getIntlPD());

  if (!hasBias_) {
    return;
  }
  biasDiff_.reset(new MkldnnBuffer());
  real* biasGradData = biases_->getWGrad()->getData();
  biasDiff_->resetUser(biasGradData, biasDims_, biasFmt_, engine_);
  biasDiff_->resetIntl(biasDiff_->getUser());
  CHECK(biasDiff_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
    << "should always be format::x, or changed in new mkldnn version";
}

void MkldnnFcLayer::resetDnnBotDiff(
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  if (!hasBotGrad()) {
    return;
  }
  CHECK(bwdDataPD) << "should have bwdDataPD";

  botDiff_.reset(new MkldnnBuffer());
  const MatrixPtr& botGrad = getInputGrad(0);
  real* botGradData = botGrad->getData();
  if (prevIsDnn_[0]) {
    botDiff_->resetUser(botGradData, bwdDataPD->diff_src_primitive_desc());
    botDiff_->resetIntl(botDiff_->getUser());
  } else {
    botDiff_->resetUser(botGradData, botDims_, botFmt_, engine_);
    botDiff_->resetIntl(bwdDataPD->diff_src_primitive_desc());
    botDiff_->resetReorder(dnnIntl2User);
  }
}

void MkldnnFcLayer::resetDnnBwdPipeline(
  const std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
  const std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD) {
  /// backward weight and bias
  CHECK(bwdWgtPD);
  CHECK(botData_->getIntl());
  CHECK(topDiffBwdWgt_->getIntl());
  CHECK(wgtDiff_->getIntl());
  if (hasBias_) {
    CHECK(biasDiff_->getIntl());
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD,
        *(botData_->getIntl()), *(topDiffBwdWgt_->getIntl()),
        *(wgtDiff_->getIntl()), *(biasDiff_->getIntl())));
  } else {
    bwdWgt_.reset(new fc_bwdWgt(*bwdWgtPD,
        *(botData_->getIntl()), *(topDiffBwdWgt_->getIntl()),
        *(wgtDiff_->getIntl())));
  }

  if (topDiffBwdWgt_->needReorder()) {
    pipelineBwd_.push_back(*topDiffBwdWgt_->getReorder());
  }
  pipelineBwd_.push_back(*bwdWgt_);
  CHECK_EQ(wgtDiff_->needReorder(), false) << "wgt should not need reorder!";
  if (hasBias_) {
    CHECK_EQ(biasDiff_->needReorder(), false)
      << "bias should not need reorder!";
  }

  /// backward data
  if (!hasBotGrad()) {
    return;
  }
  CHECK(bwdDataPD);
  bwdData_.reset(new fc_bwdData(*bwdDataPD,
    *(topDiff_->getIntl()), *(wgtData_->getIntl()),
    *(botDiff_->getIntl())));
  if (topDiff_->needReorder()) {
    pipelineBwd_.push_back(*topDiff_->getReorder());
  }
  pipelineBwd_.push_back(*bwdData_);
  if (botDiff_->needReorder()) {
    pipelineBwd_.push_back(*botDiff_->getReorder());
  }
}

void MkldnnFcLayer::backwardDnnVal() {
  real* topGradData = getOutputGrad()->getData();
  real* botValData = getInputValue(0)->getData();

  topDiffBwdWgt_->updateUserData(topGradData);
  botData_->updateUserData(botValData);

  if (hasBotGrad()) {
    real* botGradData = getInputGrad(0)->getData();
    topDiff_->updateUserData(topGradData);
    botDiff_->updateUserData(botGradData);
  }

  stream(stream::kind::eager).submit(pipelineBwd_).wait();
}


void MkldnnFcLayer::updateParameter(const UpdateCallback &callback) {
  if (weight_->getWGrad()) {
    weight_->getParameterPtr()->incUpdate(callback);
  }

  if (hasBias_) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

inline bool MkldnnFcLayer::hasBotGrad() {
  return getInputGrad(0) != nullptr ? true : false;
}



}  // namespace paddle
