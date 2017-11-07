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
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  CHECK_EQ(layerSize_, getSize()) << "this layer size can not be changed";
  reshapeInput(bs, ih, iw);
  ic = inputLayers_[0]->getSize() / ih / iw;
  CHECK_EQ((size_t)ic * ih * iw, inputLayers_[0]->getSize());
  CHECK_EQ(inputElemenCnt_, (size_t)bs * ic * ih * iw);
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    CHECK_EQ(int64_t(bs), inputLayers_[i]->getOutput().getBatchSize());
    CHECK_EQ(layerSize_, inputLayers_[i]->getSize());
  }

  oc = ic;
  oh = ih;
  ow = iw;
  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
  printSizeInfo();
}

void MKLDNNAddtoLayer::resetFwd(std::vector<primitive>& pipeline,
                                MKLDNNMatrixPtr& in,
                                MKLDNNMatrixPtr& wgt,
                                MKLDNNMatrixPtr& bias,
                                MKLDNNMatrixPtr& out) {
  if (biases_) {
    LOG(FATAL) << "not implemented yet";
  }
  resetFwdBuffers(inVals_, out);
  in = inVals_[0];

  std::shared_ptr<sum::primitive_desc> fwdPD;
  resetFwdPD(fwdPD, inVals_, out);

  resetFwdPipeline(pipeline, fwdPD, inVals_, out);
}

void MKLDNNAddtoLayer::resetBwd(std::vector<primitive>& pipeline,
                                MKLDNNMatrixPtr& in,
                                MKLDNNMatrixPtr& wgt,
                                MKLDNNMatrixPtr& bias,
                                MKLDNNMatrixPtr& out) {
  resetBwdBuffers(inGrads_, out);
  in = inGrads_[0];

  // backward only need share output grad to input grad
  for (size_t i = 0; i < inGrads_.size(); i++) {
    if (inGrads_[i] != nullptr) {
      inGrads_[i] = out;
      inputLayers_[i]->getOutputGrad()->setData(inGrads_[i]->getData());
    }
  }
}

void MKLDNNAddtoLayer::updateWeights(const UpdateCallback& callback) {
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

void MKLDNNAddtoLayer::resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
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
}

void MKLDNNAddtoLayer::resetFwdPD(std::shared_ptr<sum::primitive_desc>& pd,
                                  std::vector<MKLDNNMatrixPtr>& inputs,
                                  MKLDNNMatrixPtr out) {
  std::vector<double> scales(inputs.size(), 1.0);
  std::vector<memory::primitive_desc> srcPDs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcPDs.push_back(inputs[i]->getPrimitiveDesc());
  }
  CHECK(out);
  pd.reset(new sum::primitive_desc(out->getMemoryDesc(), scales, srcPDs));
  CHECK_PRIMITIVE_DESC_EQ(out, pd->dst_primitive_desc());
}

void MKLDNNAddtoLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<sum::primitive_desc>& pd,
    std::vector<MKLDNNMatrixPtr>& inputs,
    MKLDNNMatrixPtr& out) {
  std::vector<primitive::at> srcs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcs.push_back(*(inputs[i]));
  }
  fwd_.reset(new sum(*pd, srcs, *out));
  pipeline.push_back(*fwd_);
}

void MKLDNNAddtoLayer::resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                                       MKLDNNMatrixPtr& out) {
  CHECK(outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  CHECK(out);

  inputs.resize(inputLayers_.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    resetInGrad(inputs[i], inVal_->getPrimitiveDesc(), i);
    CHECK_PRIMITIVE_DESC_EQ(inputs[i], out->getPrimitiveDesc());
  }
}

}  // namespace paddle
