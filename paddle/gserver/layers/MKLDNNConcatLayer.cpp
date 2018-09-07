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

#include "MKLDNNConcatLayer.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_concat, MKLDNNConcatLayer);

bool MKLDNNConcatLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }
  CHECK_GT(inputLayers_.size(), 1UL);
  CHECK(!biasParameter_);
  return true;
}

void MKLDNNConcatLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);
  ic = inputLayers_[0]->getSize() / ih / iw;
  CHECK_EQ((size_t)ic * ih * iw, inputLayers_[0]->getSize());
  CHECK_EQ(inputLayers_[0]->getOutputValue()->getElementCnt(),
           (size_t)bs * ic * ih * iw);
  CHECK_GT(inputLayers_.size(), 1UL);
  channels_.resize(inputLayers_.size());
  channels_[0] = ic;
  oc = ic;
  for (size_t i = 1; i < inputLayers_.size(); i++) {
    int batchsize = 0, height = 0, witdh = 0;
    reshapeInput(batchsize, height, witdh, i);
    CHECK_EQ(bs, batchsize);
    CHECK_EQ(ih, height);
    CHECK_EQ(iw, witdh);

    channels_[i] = inputLayers_[i]->getSize() / height / witdh;
    CHECK_EQ((size_t)channels_[i] * height * witdh, inputLayers_[i]->getSize());
    oc += channels_[i];
  }
  oh = ih;
  ow = iw;
  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);
}

void MKLDNNConcatLayer::resetFwd(std::vector<primitive>& pipeline,
                                 std::vector<MKLDNNMatrixPtr>& inputs,
                                 MKLDNNMatrixPtr& out) {
  resetFwdBuffers(inputs, out);

  std::shared_ptr<concat::primitive_desc> fwdPD;
  resetFwdPD(fwdPD, inputs, out);

  resetFwdPipeline(pipeline, fwdPD, inputs, out);
}

void MKLDNNConcatLayer::resetBwd(std::vector<primitive>& pipeline,
                                 std::vector<MKLDNNMatrixPtr>& inputs,
                                 MKLDNNMatrixPtr& out) {
  resetBwdBuffers(inputs, out);

  resetBwdPipeline(pipeline, bwds_, inputs, out);
}

void MKLDNNConcatLayer::resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                                        MKLDNNMatrixPtr& out) {
  inputs.resize(inputLayers_.size());
  bool has8c = false, has16c = false, hasnc = false;
  for (size_t i = 0; i < inputs.size(); i++) {
    resetInValue(inputs[i], nullptr, i, channels_[i]);
    inputs[i]->downSpatial();
    CHECK(inputs[i]);
    auto dm = inputs[i]->getDims();
    // inputs format can be different, but ndims must equal
    CHECK(i == 0 || dm.size() == inputs[0]->getDims().size());
    CHECK_EQ(bs_, dm[0]);
    CHECK_EQ(channels_[i], dm[1]);
    if (dm.size() > 2) {
      CHECK_EQ(ih_, dm[2]);
      CHECK_EQ(iw_, dm[3]);
    }
    if (inputs[i]->getFormat() == format::nc) {
      hasnc = true;
    }
    if (inputs[i]->getFormat() == format::nChw8c) {
      has8c = true;
    }
    if (inputs[i]->getFormat() == format::nChw16c) {
      has16c = true;
    }
  }

  format outFmt;
  if (has16c && oc_ % 16 == 0) {
    outFmt = format::nChw16c;
  } else if (has8c && oc_ % 8 == 0) {
    outFmt = format::nChw8c;
  } else if (hasnc) {
    CHECK(oh_ == 1 && ow_ == 1);
    outFmt = format::nc;
  } else {
    outFmt = format::nchw;
  }
  memory::dims outDims =
      hasnc ? memory::dims{bs_, oc_} : memory::dims{bs_, oc_, oh_, ow_};
  auto outPD = MKLDNNMatrix::createPrimitiveDesc(outDims, outFmt, engine_);
  resetOutValue(out, outPD);
}

void MKLDNNConcatLayer::resetFwdPD(std::shared_ptr<concat::primitive_desc>& pd,
                                   std::vector<MKLDNNMatrixPtr>& inputs,
                                   MKLDNNMatrixPtr out) {
  std::vector<memory::primitive_desc> srcPDs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcPDs.push_back(inputs[i]->getPrimitiveDesc());
  }
  CHECK(out);
  pd.reset(new concat::primitive_desc(out->getMemoryDesc(), axis_, srcPDs));
  CHECK_PRIMITIVE_DESC_EQ(out, pd->dst_primitive_desc());
}

void MKLDNNConcatLayer::resetFwdPipeline(
    std::vector<primitive>& pipeline,
    std::shared_ptr<concat::primitive_desc>& pd,
    std::vector<MKLDNNMatrixPtr>& inputs,
    MKLDNNMatrixPtr& out) {
  std::vector<primitive::at> srcs;
  for (size_t i = 0; i < inputs.size(); i++) {
    srcs.push_back(*(inputs[i]));
  }
  fwd_.reset(new concat(*pd, srcs, *out));
  pipeline.push_back(*fwd_);
}

void MKLDNNConcatLayer::resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                                        MKLDNNMatrixPtr& out) {
  CHECK(outVal_);
  resetOutGrad(out, outVal_->getPrimitiveDesc());
  CHECK(out);

  inputs.resize(inputLayers_.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    CHECK(inVals_[i]);
    resetInGrad(inputs[i], inVals_[i]->getPrimitiveDesc(), i);
    CHECK_PRIMITIVE_DESC_EQ(inputs[i], inVals_[i]->getPrimitiveDesc());
  }
}

void MKLDNNConcatLayer::resetBwdPipeline(
    std::vector<mkldnn::primitive>& pipeline,
    std::vector<std::shared_ptr<mkldnn::primitive>>& prims,
    std::vector<MKLDNNMatrixPtr>& inputs,
    MKLDNNMatrixPtr& out) {
  // reset the backward primitives
  memory::dims offsets = {0, 0, 0, 0};
  prims.resize(inputs.size());
  CHECK_EQ(inputs.size(), channels_.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto viewPD = view::primitive_desc(
        out->getPrimitiveDesc(), inputs[i]->getDims(), offsets);
    auto bwdPD = reorder::primitive_desc(viewPD.dst_primitive_desc(),
                                         inputs[i]->getPrimitiveDesc());
    prims[i].reset(new reorder(bwdPD, *out, *(inputs[i])));
    offsets[axis_] += channels_[i];
    // push to pipeline
    pipeline.push_back(*prims[i]);
  }
}

}  // namespace paddle
