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

#include "MKLDNNLRNLayer.h"
#include "paddle/utils/Logging.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_lrn, MKLDNNLRNLayer);

bool MKLDNNLRNLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);
  const NormConfig& conf = config_.inputs(0).norm_conf();
  localSize_ = conf.size();
  alpha_ = conf.scale();
  beta_ = conf.pow();

  ic_ = conf.channels();
  oc_ = ic_;
  iw_ = conf.img_size();
  ow_ = conf.output_x();
  ih_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  oh_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  CHECK_EQ(iw_, ow_);
  CHECK_EQ(ih_, oh_);
  return true;
}

void MKLDNNLRNLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  CHECK_EQ(inputLayers_.size(), 1UL);
  reshapeInput(bs, ih, iw);
  // ic_ and oc can not be changed
  CHECK_EQ((size_t)ic,
           inputLayers_[0]->getOutputValue()->getElementCnt() / bs / ih / iw)
      << "Input channel can not be changed";
  oh = ih;
  ow = iw;
  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNLRNLayer::resetFwd(std::vector<primitive>& pipeline,
                              std::vector<MKLDNNMatrixPtr>& inputs,
                              MKLDNNMatrixPtr& out) {
  resetFwdBuffers(inputs[0], out);

  resetFwdPD(fwdPD_, inputs[0], out);

  resetFwdPipeline(pipeline, fwdPD_, inputs[0], out);
}

void MKLDNNLRNLayer::resetBwd(std::vector<primitive>& pipeline,
                              std::vector<MKLDNNMatrixPtr>& inputs,
                              MKLDNNMatrixPtr& out) {
  std::shared_ptr<lrn_bwd::primitive_desc> pd;

  resetBwdBuffers(inputs[0], out);

  resetBwdPD(pd, inputs[0], out);

  resetBwdPipeline(pipeline, pd, inputs[0], out);
}

void MKLDNNLRNLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                     MKLDNNMatrixPtr& out) {
  resetInValue(in);
  CHECK(in);
  resetOutValue(out, in->getPrimitiveDesc());
}

void MKLDNNLRNLayer::resetFwdPD(std::shared_ptr<lrn_fwd::primitive_desc>& pd,
                                MKLDNNMatrixPtr in,
                                MKLDNNMatrixPtr out) {
  prop_kind pk = passType_ == PASS_TEST ? prop_kind::forward_scoring
                                        : prop_kind::forward_training;
  auto fwdDesc = lrn_fwd::desc(pk,
                               algorithm::lrn_across_channels,
                               in->getMemoryDesc(),
                               localSize_,
                               alpha_,
                               beta_,
                               1.0f);
  pd.reset(new lrn_fwd::primitive_desc(fwdDesc, engine_));
  // prepare workspace if necessary
  workspace_ =
      passType_ != PASS_TEST
          ? std::make_shared<memory>(memory(pd->workspace_primitive_desc()))
          : nullptr;
}

void MKLDNNLRNLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<lrn_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {
  fwd_ = workspace_
             ? std::make_shared<lrn_fwd>(lrn_fwd(*pd, *in, *workspace_, *out))
             : std::make_shared<lrn_fwd>(lrn_fwd(*pd, *in, *out));
  pipeline.push_back(*fwd_);
}

void MKLDNNLRNLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                     MKLDNNMatrixPtr& out) {
  CHECK(inVals_[0] && outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  resetInGrad(in, inVals_[0]->getPrimitiveDesc());
}

void MKLDNNLRNLayer::resetBwdPD(std::shared_ptr<lrn_bwd::primitive_desc>& pd,
                                MKLDNNMatrixPtr& in,
                                MKLDNNMatrixPtr& out) {
  pd = nullptr;
  if (in == nullptr) {
    return;
  }
  CHECK(out);
  auto bwdDesc = lrn_bwd::desc(algorithm::lrn_across_channels,
                               in->getMemoryDesc(),
                               out->getMemoryDesc(),
                               localSize_,
                               alpha_,
                               beta_,
                               1.0f);
  pd.reset(new lrn_bwd::primitive_desc(bwdDesc, engine_, *fwdPD_));
}

void MKLDNNLRNLayer::resetBwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<lrn_bwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {
  if (pd == nullptr) {
    return;
  }
  CHECK(inVals_[0]);
  CHECK(workspace_);
  bwdData_ = std::make_shared<lrn_bwd>(
      lrn_bwd(*pd, *inVals_[0], *out, *workspace_, *in));
  pipeline.push_back(*bwdData_);
}

}  // namespace paddle
