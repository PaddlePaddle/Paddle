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

#include "MKLDNNPoolLayer.h"
#include "paddle/utils/Logging.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

REGISTER_LAYER(mkldnn_pool, MKLDNNPoolLayer);

bool MKLDNNPoolLayer::init(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  return true;
}

void MKLDNNPoolLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  reshapeInput(bs, ih, iw);

  // cal output sizes
  // oc can not be changed

  reshapeOutput(oh, ow);
  resizeOutput(bs, oc * oh * ow);

  printSizeInfo();
}

void MKLDNNPoolLayer::resetFwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  resetFwdBuffers(in, out);

  resetFwdPD(fwdPD_, in, out);

  resetFwdPipeline(pipeline, fwdPD_, in, out);

  printValueFormatFlow();
}

void MKLDNNPoolLayer::resetBwd(std::vector<primitive>& pipeline,
                               MKLDNNMatrixPtr& in,
                               MKLDNNMatrixPtr& wgt,
                               MKLDNNMatrixPtr& bias,
                               MKLDNNMatrixPtr& out) {
  std::shared_ptr<pool_bwd::primitive_desc> pd;

  resetBwdBuffers(in, out);

  resetBwdPD(pd, in, out);

  resetBwdPipeline(pipeline, pd, in, out);

  printGradFormatFlow();
}

void MKLDNNPoolLayer::updateInputData() {
  inVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
}

void MKLDNNPoolLayer::resetFwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  resetInValue(in);
  resetOutValue(out);
}

void MKLDNNPoolLayer::resetInValue(MKLDNNMatrixPtr& in) {}

void MKLDNNPoolLayer::resetOutValue(MKLDNNMatrixPtr& out) {}

void MKLDNNPoolLayer::resetFwdPD(std::shared_ptr<pool_fwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr in,
                                 MKLDNNMatrixPtr out) {}

void MKLDNNPoolLayer::resetFwdPipeline(
    std::vector<mkldnn::primitive>& pipeline,
    std::shared_ptr<pool_fwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {}

void MKLDNNPoolLayer::resetBwdBuffers(MKLDNNMatrixPtr& in,
                                      MKLDNNMatrixPtr& out) {
  resetOutGrad(out);
  resetInGrad(in);
}
void MKLDNNPoolLayer::resetOutGrad(MKLDNNMatrixPtr& out) {}

void MKLDNNPoolLayer::resetInGrad(MKLDNNMatrixPtr& in) {}

void MKLDNNPoolLayer::resetBwdPD(std::shared_ptr<pool_bwd::primitive_desc>& pd,
                                 MKLDNNMatrixPtr& in,
                                 MKLDNNMatrixPtr& out) {}

void MKLDNNPoolLayer::resetBwdPipeline(
    std::vector<mkldnn::primitive>& pipeline,
    std::shared_ptr<pool_bwd::primitive_desc>& pd,
    MKLDNNMatrixPtr& in,
    MKLDNNMatrixPtr& out) {}

}  // namespace paddle
