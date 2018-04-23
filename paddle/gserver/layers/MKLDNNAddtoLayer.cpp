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

#include "MKLDNNAddtoLayer.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_addto, MKLDNNAddtoLayer);

bool MKLDNNAddtoLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  layerSize_ = getSize();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK_EQ(layerSize_, inputLayers_[i]->getSize()) << "input size must equal";
  }
  if (biasParameter_.get() != NULL) {
    biases_ =
        std::unique_ptr<Weight>(new Weight(1, layerSize_, biasParameter_, 0));
  }
  return true;
}

void MKLDNNAddtoLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  CHECK_EQ(layerSize_, getSize()) << "this layer size can not be changed";
  reshapeInput(bs, ih, iw);
  ic = inputLayers_[0]->getSize() / ih / iw;
  CHECK_EQ((size_t)ic * ih * iw, inputLayers_[0]->getSize());
  CHECK_EQ(inputLayers_[0]->getOutputValue()->getElementCnt(),
           (size_t)bs * ic * ih * iw);
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK_EQ(int64_t(bs), inputLayers_[i]->getOutput().getBatchSize());
    CHECK_EQ(layerSize_, inputLayers_[i]->getSize());
  }

  oc = ic;
  oh = ih;
  ow = iw;
  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNAddtoLayer::resetFwd(std::vector<primitive>& pipeline,
                                std::vector<MKLDNNMatrixPtr>& inputs,
                                MKLDNNMatrixPtr& out) {
  resetFwdBuffers(inputs, biasVal_, out);

  std::shared_ptr<sum::primitive_desc> fwdPD;
  std::shared_ptr<sum::primitive_desc> biasPD;
  resetFwdPD(fwdPD, biasPD, inputs, biasVal_, out);

  resetFwdPipeline(pipeline, fwdPD, biasPD, inputs, biasVal_, out);
}

void MKLDNNAddtoLayer::resetBwd(std::vector<primitive>& pipeline,
                                std::vector<MKLDNNMatrixPtr>& inputs,
                                MKLDNNMatrixPtr& out) {
  resetBwdBuffers(inputs, biasGrad_, out);

  // backward only need share output grad to input grad
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      inputs[i] = out;
      inputLayers_[i]->getOutputGrad()->setData(inputs[i]->getData());
    }
  }

  // backward bias
  bwdBias_ = nullptr;
  if (biasGrad_) {
    std::vector<float> scales(bs_, 1.0);
    std::vector<memory::primitive_desc> srcPDs(bs_,
                                               biasGrad_->getPrimitiveDesc());
    auto biasPD =
        sum::primitive_desc(biasGrad_->getMemoryDesc(), scales, srcPDs);
    std::vector<primitive::at> srcs;
    for (size_t i = 0; i < grads_.size(); ++i) {
      srcs.push_back(*(grads_[i]));
    }
    bwdBias_.reset(new sum(biasPD, srcs, *biasGrad_));
    pipeline.push_back(*bwdBias_);
  }
}

void MKLDNNAddtoLayer::updateWeights(const UpdateCallback& callback) {
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

void MKLDNNAddtoLayer::prepareBias(MKLDNNMatrixPtr& bias,
                                   const MatrixPtr& biasMat,
                                   const MKLDNNMatrixPtr& out,
                                   std::vector<MKLDNNMatrixPtr>& outs) {
  auto pd = MKLDNNMatrix::createPrimitiveDesc(
      {(int)layerSize_}, memory::format::x, engine_);
  bias = MKLDNNMatrix::create(pd, biasMat);
  outs.clear();
  real* data = out->getData();
  CHECK_EQ(bs_ * layerSize_, out->getElementCnt());
  for (int i = 0; i < bs_; ++i) {
    MatrixPtr tmp =
        Matrix::create(data + i * layerSize_, 1, layerSize_, false, false);
    outs.push_back(MKLDNNMatrix::create(bias->getPrimitiveDesc(), tmp));
  }
}

void MKLDNNAddtoLayer::resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                                       MKLDNNMatrixPtr& bias,
                                       MKLDNNMatrixPtr& out) {
  inputs.resize(inputLayers_.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    resetInValue(inputs[i], nullptr, i);
    CHECK(inputs[i]);
    inputs[i]->downSpatial();
  }
  for (size_t i = 1; i < inputs.size(); i++) {
    CHECK_PRIMITIVE_DESC_EQ(inputs[i], inputs[0]->getPrimitiveDesc());
  }

  resetOutValue(out, inputs[0]->getPrimitiveDesc());

  if (biases_ && biases_->getW()) {
    prepareBias(bias, biases_->getW(), out, vals_);
  } else {
    bias = nullptr;
  }
}

void MKLDNNAddtoLayer::resetFwdPD(std::shared_ptr<sum::primitive_desc>& pd,
                                  std::shared_ptr<sum::primitive_desc>& biasPD,
                                  std::vector<MKLDNNMatrixPtr>& inputs,
                                  MKLDNNMatrixPtr bias,
                                  MKLDNNMatrixPtr out) {
  std::vector<float> scales(inputs.size(), 1.0);
  std::vector<memory::primitive_desc> srcPDs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcPDs.push_back(inputs[i]->getPrimitiveDesc());
  }
  CHECK(out);
  pd.reset(new sum::primitive_desc(out->getMemoryDesc(), scales, srcPDs));
  CHECK_PRIMITIVE_DESC_EQ(out, pd->dst_primitive_desc());

  biasPD = nullptr;
  if (bias) {
    std::vector<float> scales(2, 1.0);
    std::vector<memory::primitive_desc> srcPDs(2, bias->getPrimitiveDesc());
    biasPD.reset(
        new sum::primitive_desc(bias->getMemoryDesc(), scales, srcPDs));
    CHECK_PRIMITIVE_DESC_EQ(bias, biasPD->dst_primitive_desc());
  }
}

void MKLDNNAddtoLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<sum::primitive_desc>& pd,
    std::shared_ptr<sum::primitive_desc>& biasPD,
    std::vector<MKLDNNMatrixPtr>& inputs,
    MKLDNNMatrixPtr& bias,
    MKLDNNMatrixPtr& out) {
  std::vector<primitive::at> srcs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcs.push_back(*(inputs[i]));
  }
  fwd_.reset(new sum(*pd, srcs, *out));
  pipeline.push_back(*fwd_);

  fwdBias_.clear();
  if (biasPD == nullptr || bias == nullptr) {
    return;
  }
  fwdBias_.resize(vals_.size());
  for (size_t i = 0; i < vals_.size(); ++i) {
    std::vector<primitive::at> srcs;
    srcs.push_back(*(vals_[i]));
    srcs.push_back(*bias);
    fwdBias_[i].reset(new sum(*biasPD, srcs, *vals_[i]));
    pipeline.push_back(*fwdBias_[i]);
  }
}

void MKLDNNAddtoLayer::resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                                       MKLDNNMatrixPtr& bias,
                                       MKLDNNMatrixPtr& out) {
  CHECK(outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  CHECK(out);

  inputs.resize(inputLayers_.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    resetInGrad(inputs[i], inVals_[i]->getPrimitiveDesc(), i);
    CHECK_PRIMITIVE_DESC_EQ(inputs[i], out->getPrimitiveDesc());
  }

  if (biases_ && biases_->getWGrad()) {
    prepareBias(bias, biases_->getWGrad(), out, grads_);
  } else {
    bias = nullptr;
  }
}

}  // namespace paddle
