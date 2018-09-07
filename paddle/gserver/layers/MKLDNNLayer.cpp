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

#include "MKLDNNLayer.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

bool MKLDNNLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  CHECK(FLAGS_use_mkldnn) << "MKLDNNLayers only support use_mkldnn."
                          << "Please set WITH_MKL=ON "
                          << "and set use_mkldnn=True";
  CHECK(!useGpu_) << "Do not support GPU yet";

  // set device id before Layer::init
  setDevice(MKLDNN_DEVICE);
  // change param device to MKLDNN device
  setParamsDevice(MKLDNN_DEVICE, parameterMap);
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }
  setOutputMap();
  checkCPUOutputsNumber();

  stream_.reset(new MKLDNNStream());
  engine_ = CPUEngine::Instance().getEngine();
  return true;
}

void MKLDNNLayer::forward(PassType passType) {
  passType_ = passType;

  {
    REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
    CHECK(!inputLayers_.empty());
    copySeqInfoToOutputs();
    if (condition_ != keepCondition()) {
      VLOG(MKLDNN_BASE) << getName() << " reset mkldnn forward";
      condition_ = keepCondition();
      reshape(bs_, ic_, ih_, iw_, oc_, oh_, ow_);
      printSizeInfo();
      // the output_.value and output_.grad are shared with CPU device
      shareCPUDevice();
      pipelineFwd_.clear();
      inVals_.resize(inputLayers_.size(), nullptr);
      extInVals_.resize(inputLayers_.size(), nullptr);
      cvtInVals_.resize(inputLayers_.size(), nullptr);
      resetFwd(pipelineFwd_, inVals_, outVal_);
      prepareValueConversions(pipelineFwd_);
      convertWeightsFromPaddle();
      printValueFormat();
      needResetBwd_ = true;
    }

    if (inputLayers_[0]->getType() == "data" && inputLayers_.size() == 1) {
      // Update input value data when input layer is "data" type,
      // since the input value data address might be changed.
      CHECK(extInVals_[0]);
      extInVals_[0]->setData(getInputValue(0, CPU_DEVICE)->getData());
    }

    if (!outputOnlyMKLDNN_) {
      clearGrads();
    }
    stream_->submit(pipelineFwd_);
  }
  {
    REGISTER_TIMER_INFO("FwActTimer", getName().c_str());
    forwardActivation();
  }
}

void MKLDNNLayer::backward(const UpdateCallback& callback) {
  if (needResetBwd_) {
    VLOG(MKLDNN_BASE) << getName() << " reset mkldnn backward";
    pipelineBwd_.clear();
    inGrads_.resize(inputLayers_.size(), nullptr);
    extInGrads_.resize(inputLayers_.size(), nullptr);
    cvtInGrads_.resize(inputLayers_.size(), nullptr);
    pipelineMergeGrad_.clear();
    mergeGrad_ = nullptr;
    resetBwd(pipelineBwd_, inGrads_, outGrad_);
    prepareGradConversions(pipelineBwd_);
    printGradFormat();
    needResetBwd_ = false;
  }

  // merge grad must before backward activation
  if (mergeGrad_) {
    REGISTER_TIMER_INFO("MergeBpGrad", getName().c_str());
    stream_->submit(pipelineMergeGrad_);
  }
  {
    REGISTER_TIMER_INFO("BpActTimer", getName().c_str());
    backwardActivation();
  }
  {
    REGISTER_TIMER_INFO("mkldnn_bwdTimer", getName().c_str());
    stream_->submit(pipelineBwd_);
  }
  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    updateWeights(callback);
  }
}

void MKLDNNLayer::reshapeInput(int& batchsize,
                               int& height,
                               int& width,
                               size_t idx) {
  const Argument& input = inputLayers_[idx]->getOutput();
  batchsize = input.getBatchSize();
  int h = input.getFrameHeight();
  int w = input.getFrameWidth();
  if (h != 0) {
    height = h;
  }
  if (w != 0) {
    width = w;
  }
  height = height != 0 ? height : 1;
  width = width != 0 ? width : 1;
}

void MKLDNNLayer::reshapeOutput(size_t height, size_t width) {
  output_.setFrameHeight(height);
  output_.setFrameWidth(width);
  for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
    outputOtherDevice_[i].setFrameHeight(height);
    outputOtherDevice_[i].setFrameWidth(width);
  }
}

void MKLDNNLayer::resetWithMatrix(MKLDNNMatrixPtr& dnn,
                                  const MatrixPtr& mat,
                                  memory::primitive_desc pd) {
  dnn = nullptr;
  if (mat == nullptr) {
    return;
  }
  dnn = MKLDNNMatrix::create(pd, mat);
}

void MKLDNNLayer::resetInValue(
    MKLDNNMatrixPtr& in,
    const std::shared_ptr<memory::primitive_desc>& intPD,
    size_t idx,
    int inputChannel) {
  cvtInVals_[idx] = nullptr;
  extInVals_[idx] = nullptr;
  in = nullptr;
  inputChannel = inputChannel == 0 ? ic_ : inputChannel;
  CHECK_GT(bs_ * inputChannel * ih_ * iw_, 0);
  auto extPD = MKLDNNMatrix::createPrimitiveDesc(
      {bs_, inputChannel, ih_, iw_}, format::nchw, engine_);
  const MatrixPtr& inMat = inputLayers_[idx]->getOutputValue();
  extInVals_[idx] = std::dynamic_pointer_cast<MKLDNNMatrix>(inMat);
  CHECK_EQ(inputIsOnlyMKLDNN(), extInVals_[idx] != nullptr);
  if (extInVals_[idx] == nullptr ||
      extInVals_[idx]->getFormat() == format::nc) {
    extInVals_[idx] = MKLDNNMatrix::create(extPD, inMat);
  }
  in = extInVals_[idx];
  if (nullptr == intPD || in->getPrimitiveDesc() == *intPD) {
    return;
  }
  // need create reorder
  in = MKLDNNMatrix::create(*intPD);
  cvtInVals_[idx] = MKLDNNMatrix::createReorder(extInVals_[idx], in);
  CHECK(cvtInVals_[idx]) << "should not be emptry";
}

void MKLDNNLayer::resetOutValue(MKLDNNMatrixPtr& out,
                                memory::primitive_desc intPD) {
  cvtOutVal_ = nullptr;
  out = MKLDNNMatrix::create(intPD, output_.value);
  extOutVal_ = out;
  if (outputIsOnlyMKLDNN() || isPaddleFormat(extOutVal_->getFormat())) {
    return;
  }
  // need create reorder
  CHECK_GT(bs_ * oc_ * oh_ * ow_, 0);
  extOutVal_ = MKLDNNMatrix::create(
      memory::dims{bs_, oc_, oh_, ow_}, format::nchw, engine_, output_.value);
  out = MKLDNNMatrix::create(intPD);
  cvtOutVal_ = MKLDNNMatrix::createReorder(out, extOutVal_);
  CHECK(cvtOutVal_) << "should not be empty";
}

void MKLDNNLayer::resetInGrad(MKLDNNMatrixPtr& in,
                              memory::primitive_desc intPD,
                              size_t idx) {
  cvtInGrads_[idx] = nullptr;
  extInGrads_[idx] = nullptr;
  in = nullptr;
  LayerPtr& input = inputLayers_[idx];
  if (input->getOutputGrad() == nullptr) {
    // no need input grad
    return;
  }
  CHECK(inputIsOnlyMKLDNN() || input->getOutputMapSize() <= 1)
      << "only support input is MKLDNN layer or only have one output layer";
  // when input is a mkldnn branch node,
  // this layer will save input grad to a internal buffer,
  // and the mkldnn input layer will merge them to actual prev->output_.grad
  const MatrixPtr& inMat =
      input->getOutputMapSize() <= 1 ? input->getOutputGrad() : nullptr;
  in = MKLDNNMatrix::create(intPD, inMat);
  Argument& arg = input->getOutput(this->getName());
  arg.grad = std::dynamic_pointer_cast<Matrix>(in);
  CHECK_PRIMITIVE_DESC_EQ(inVals_[idx], intPD);
  if (inputIsOnlyMKLDNN()) {
    return;
  }

  extInGrads_[idx] = in;
  if (isPaddleFormat(extInGrads_[idx]->getFormat())) {
    return;
  }
  // need create reorder
  CHECK(extInVals_[idx] != nullptr &&
        isPaddleFormat(extInVals_[idx]->getFormat()))
      << "should have external input value and the format must be nchw(nc)";
  extInGrads_[idx] =
      MKLDNNMatrix::create(extInVals_[idx]->getPrimitiveDesc(), inMat);
  CHECK_PRIMITIVE_DESC_EQ(inVals_[idx], intPD);
  in = MKLDNNMatrix::create(intPD);
  cvtInGrads_[idx] = MKLDNNMatrix::createReorder(in, extInGrads_[idx]);
  CHECK(cvtInGrads_[idx]);
}

void MKLDNNLayer::resetOutGrad(MKLDNNMatrixPtr& out,
                               memory::primitive_desc intPD) {
  cvtOutGrad_ = nullptr;
  extOutGrad_ = nullptr;
  out = nullptr;
  MatrixPtr& outMat = output_.grad;
  out = MKLDNNMatrix::create(intPD, outMat);
  resetMergeGrad(out);
  if (outputIsOnlyMKLDNN()) {
    return;
  }
  CHECK_LE(outputMap_.size(), 1U) << "do not support mixed with cpu device";
  extOutGrad_ = out;
  if (isPaddleFormat(extOutGrad_->getFormat())) {
    return;
  }
  // need create reorder
  CHECK(extOutVal_ != nullptr && isPaddleFormat(extOutVal_->getFormat()))
      << "should have external output value and the format must be nchw(nc)";
  extOutGrad_ = MKLDNNMatrix::create(extOutVal_->getPrimitiveDesc(), outMat);
  CHECK_PRIMITIVE_DESC_EQ(outVal_, intPD);
  out = MKLDNNMatrix::create(intPD);
  cvtOutGrad_ = MKLDNNMatrix::createReorder(extOutGrad_, out);
  CHECK(cvtOutGrad_);
}

void MKLDNNLayer::resetMergeGrad(MKLDNNMatrixPtr& out) {
  mergeGrad_ = nullptr;
  pipelineMergeGrad_.clear();
  if (outputMap_.size() <= 1 || !outputIsOnlyMKLDNN()) {
    // do not merge when output is not all MKLDNN or only one output
    return;
  }
  CHECK(out) << "should have reset internal ouput grad";
  std::vector<float> scales(outputMap_.size(), 1.0);
  std::vector<memory::primitive_desc> srcPDs;
  std::vector<primitive::at> srcs;
  for (auto it = outputMap_.begin(); it != outputMap_.end(); ++it) {
    MKLDNNMatrixPtr src =
        std::dynamic_pointer_cast<MKLDNNMatrix>(it->second->grad);
    CHECK(src) << "should be MKLDNNMatrix";
    auto srcDims = src->getDims();
    auto dstDims = out->getDims();
    CHECK_EQ(srcDims.size(), dstDims.size());
    for (size_t i = 0; i < srcDims.size(); ++i) {
      CHECK_EQ(srcDims[i], dstDims[i]);
    }
    VLOG(MKLDNN_BASE) << getName() << " has output grad " << it->first
                      << ", format " << src->getFormat();
    srcPDs.push_back(src->getPrimitiveDesc());
    srcs.push_back(*src);
  }

  auto sumPD = sum::primitive_desc(out->getMemoryDesc(), scales, srcPDs);
  mergeGrad_.reset(new sum(sumPD, srcs, *out));
  pipelineMergeGrad_.insert(pipelineMergeGrad_.begin(), *mergeGrad_);
}

}  // namespace paddle
