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

#include "MKLDNNLayer.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;

namespace paddle {

bool MKLDNNLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  CHECK(FLAGS_use_mkldnn) << "MkldnnLayers only support use_mkldnn."
                          << "Please set WITH_MKLDNN=ON "
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
    size_t elemenCnt = inputLayers_[0]->getOutputValue()->getElementCnt();
    if (inputElemenCnt_ != elemenCnt) {
      VLOG(MKLDNN_BASE) << getName() << " reset mkldnn forward";
      // reset when input total sizes changed, not only the batchsize
      inputElemenCnt_ = elemenCnt;
      pipelineFwd_.clear();
      reshape(bs_, ic_, ih_, iw_, oc_, oh_, ow_);
      // all cpu device output grad or value share output's
      shareCPUDevice();
      resetFwd(pipelineFwd_, inVal_, wgtVal_, biasVal_, outVal_);
      // MKLDNNLayer output value should be MKLDNNMatrix
      // so external output value is necessary.
      // Then external input value is not necessary,
      // since input may be mkldnn internal buffer.
      CHECK(extOutVal_) << "external output value is necessary";
      output_.value = std::dynamic_pointer_cast<Matrix>(extOutVal_);
      CHECK(inVal_ && outVal_) << "internal memories are necessary";
      if (cvtInVal_) {
        pipelineFwd_.insert(pipelineFwd_.begin(), *cvtInVal_);
      }
      if (cvtOutVal_) {
        pipelineFwd_.push_back(*cvtOutVal_);
      }
      convertWeightsFromPaddle();
      printSizeInfo();
      printValueFormat();
      needResetBwd_ = true;
    }

    if (inputLayers_[0]->getType() == "data" && inputLayers_.size() == 1) {
      // Update input value data when input layer is "data" type,
      // since the input value data address might be changed.
      CHECK(extInVal_);
      extInVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
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
    pipelineMergeGrad_.clear();
    mergeGrad_ = nullptr;
    resetBwd(pipelineBwd_, inGrad_, wgtGrad_, biasGrad_, outGrad_);
    // external output grad is not necessary
    // since output may be mkldnn internal buffer or merge them directly.
    CHECK(outGrad_) << "internal output grad is necessary";
    if (extOutGrad_) {
      CHECK_EQ(extOutGrad_->getData(), output_.grad->getData())
          << "the external buffer should share the same data with output_.grad";
    }
    if (cvtOutGrad_) {
      pipelineBwd_.insert(pipelineBwd_.begin(), *cvtOutGrad_);
    }
    if (cvtInGrad_) {
      pipelineBwd_.push_back(*cvtInGrad_);
    }
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

void MKLDNNLayer::reshapeInput(int& batchsize, int& height, int& width) {
  const Argument& input = inputLayers_[0]->getOutput();
  batchsize = input.getBatchSize();
  int h = input.getFrameHeight();
  int w = input.getFrameWidth();
  if (h != 0) {
    height = h;
  }
  if (w != 0) {
    width = w;
  }
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
    size_t inputIdx) {
  cvtInVal_ = nullptr;
  extInVal_ = nullptr;
  in = nullptr;
  CHECK_GT(bs_ * ic_ * ih_ * iw_, 0);
  auto extPD = MKLDNNMatrix::createPrimitiveDesc(
      {bs_, ic_, ih_, iw_}, format::nchw, engine_);
  const MatrixPtr& inMat = inputLayers_[inputIdx]->getOutputValue();
  extInVal_ = std::dynamic_pointer_cast<MKLDNNMatrix>(inMat);
  CHECK_EQ(inputIsOnlyMKLDNN(), extInVal_ != nullptr);
  if (extInVal_ == nullptr || extInVal_->getFormat() == format::nc) {
    extInVal_ = MKLDNNMatrix::create(extPD, inMat);
  }
  in = extInVal_;
  if (nullptr == intPD || in->getPrimitiveDesc() == *intPD) {
    return;
  }
  // need create reorder
  in = MKLDNNMatrix::create(*intPD);
  cvtInVal_ = MKLDNNMatrix::createReorder(extInVal_, in);
  CHECK(cvtInVal_) << "should not be emptry";
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
                              size_t inputIdx) {
  cvtInGrad_ = nullptr;
  extInGrad_ = nullptr;
  in = nullptr;
  LayerPtr& input = inputLayers_[inputIdx];
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
  CHECK_PRIMITIVE_DESC_EQ(inVal_, intPD);
  if (inputIsOnlyMKLDNN()) {
    return;
  }

  extInGrad_ = in;
  if (isPaddleFormat(extInGrad_->getFormat())) {
    return;
  }
  // need create reorder
  CHECK(extInVal_ != nullptr && isPaddleFormat(extInVal_->getFormat()))
      << "should have external input value and the format must be nchw(nc)";
  extInGrad_ = MKLDNNMatrix::create(extInVal_->getPrimitiveDesc(), inMat);
  CHECK_PRIMITIVE_DESC_EQ(inVal_, intPD);
  in = MKLDNNMatrix::create(intPD);
  cvtInGrad_ = MKLDNNMatrix::createReorder(in, extInGrad_);
  CHECK(cvtInGrad_);
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

  // TODO(TJ): remove me when mkldnn sum support different formats
  for (size_t i = 1; i < srcPDs.size(); ++i) {
    CHECK(srcPDs[0] == srcPDs[i]);
  }
  tmpOutGrad_ = out;
  tmpCvt_ = nullptr;
  if (out->getPrimitiveDesc() != srcPDs[0]) {
    tmpOutGrad_ = MKLDNNMatrix::create(srcPDs[0]);
    tmpCvt_ = MKLDNNMatrix::createReorder(tmpOutGrad_, out);
    CHECK(tmpCvt_);
    pipelineMergeGrad_.push_back(*tmpCvt_);
  }

  auto sumPD =
      sum::primitive_desc(tmpOutGrad_->getMemoryDesc(), scales, srcPDs);
  mergeGrad_.reset(new sum(sumPD, srcs, *tmpOutGrad_));
  pipelineMergeGrad_.insert(pipelineMergeGrad_.begin(), *mergeGrad_);
}

}  // namespace paddle
