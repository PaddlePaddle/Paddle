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

#pragma once

#include <vector>
#include "Layer.h"
#include "MKLDNNBase.h"
#include "mkldnn.hpp"
#include "paddle/math/MKLDNNMatrix.h"
#include "paddle/utils/Stat.h"

DECLARE_bool(use_mkldnn);

namespace paddle {

class MKLDNNLayer;
typedef std::shared_ptr<MKLDNNLayer> MKLDNNLayerPtr;

/**
 * @brief Base class of MKLDNNlayer.
 *
 */
class MKLDNNLayer : public Layer {
protected:
  // input value element count
  size_t inputElemenCnt_;
  // batch size
  int bs_;
  // input image channel, height and width
  int ic_, ih_, iw_;
  // output image channel, height and width
  int oc_, oh_, ow_;

  // backward also need reset after reset forward handle
  bool needResetBwd_;

  // is output only mkldnn
  bool outputOnlyMKLDNN_;

  // mkldnn engine, stream and primivtives
  mkldnn::engine engine_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwdWgt_;
  std::shared_ptr<mkldnn::primitive> bwdData_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

  /// value and grad are seperate as internal and external buffers.
  /// each MKLDNNLayer must init or reset internal buffer at least,
  /// and the external buffer format is always nchw of nc(when h==w==1),
  /// which is the same format as paddle.
  /// When mixed with cpu device, the output_.value and output_.grad
  /// always save the external data.
  /// When all layers are all mkldnn layers, they could be internal data.
  /// below MKLDNNMatrix buffers are all internal buffers
  MKLDNNMatrixPtr inVal_;
  MKLDNNMatrixPtr inGrad_;
  MKLDNNMatrixPtr outVal_;
  MKLDNNMatrixPtr outGrad_;
  // below are external value and grad
  MKLDNNMatrixPtr extInVal_;
  MKLDNNMatrixPtr extInGrad_;
  MKLDNNMatrixPtr extOutVal_;
  MKLDNNMatrixPtr extOutGrad_;
  // convert handle between external and internal buffers
  std::shared_ptr<mkldnn::reorder> cvtInVal_;
  std::shared_ptr<mkldnn::reorder> cvtInGrad_;
  std::shared_ptr<mkldnn::reorder> cvtOutVal_;
  std::shared_ptr<mkldnn::reorder> cvtOutGrad_;

  // weight and bias are always internal buffers
  MKLDNNMatrixPtr wgtVal_;
  MKLDNNMatrixPtr wgtGrad_;
  MKLDNNMatrixPtr biasVal_;
  MKLDNNMatrixPtr biasGrad_;

  // merge grad primitive
  std::shared_ptr<mkldnn::primitive> mergeGrad_;
  std::vector<mkldnn::primitive> pipelineMergeGrad_;
  // tmp input argument to save input grad, only used to merge grad
  Argument tmpInArg_;
  // since mkldnn sum do not support different formats:
  // can refer to https://github.com/01org/mkl-dnn/issues/134
  // so need create reorder manually and save tmp MKLDNNMatrix
  MKLDNNMatrixPtr tmpOutGrad_;
  std::shared_ptr<mkldnn::primitive> tmpCvt_;

public:
  explicit MKLDNNLayer(const LayerConfig& config)
      : Layer(config),
        inputElemenCnt_(0),
        bs_(0),
        ic_(0),
        ih_(0),
        iw_(0),
        oc_(0),
        oh_(0),
        ow_(0),
        needResetBwd_(true),
        outputOnlyMKLDNN_(false),
        engine_(mkldnn::engine::cpu, 0),
        stream_(nullptr),
        fwd_(nullptr),
        bwdWgt_(nullptr),
        bwdData_(nullptr) {}

  ~MKLDNNLayer() {}

  virtual bool init(const LayerMap& layerMap,
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

  void forward(PassType passType) override {
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
        // then external input value is not necessary,
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
        printValueFormat();
        needResetBwd_ = true;
      }

      if (inputLayers_[0]->getType() == "data") {
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

  void backward(const UpdateCallback& callback) override {
    if (needResetBwd_) {
      VLOG(MKLDNN_BASE) << getName() << " reset mkldnn backward";
      pipelineBwd_.clear();
      pipelineMergeGrad_.clear();
      mergeGrad_ = nullptr;
      resetBwd(pipelineBwd_, inGrad_, wgtGrad_, biasGrad_, outGrad_);
      // external output grad is not necessary
      // since output may be mkldnn internal buffer or merge them directly.
      CHECK(outGrad_) << "internal output grad is necessary";
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

  /**
   * reshape the input image sizes
   * and reset output image and buffer size
   * output channel can not be changed
   */
  virtual void reshape(
      int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) = 0;

  /**
   * reset the mkldnn forward primitve and memories
   * only would be called when input size changes
   */
  virtual void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out) = 0;

  /**
   * reset the mkldnn backward primitve and memories
   * only would be called when needed
   */
  virtual void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out) = 0;

  /**
   * Update weights and biases if necessary.
   */
  virtual void updateWeights(const UpdateCallback& callback) {}

  /**
   * convert weight from paddle format to mkldnn format
   * weight_ will be override
   */
  virtual void convertWeightsFromPaddle() {}

  /**
   * convert mkldnn weight to paddle format
   * weight_ will be override
   */
  virtual void convertWeightsToPaddle() {}

  /**
   * add this interface as public for unit test
   */
  void addOutputArgument(int deviceId) { Layer::addOutputArgument(deviceId); }

protected:
  /**
   * reshape the input image sizes and input batchsize
   */
  virtual void reshapeInput(int& batchsize, int& height, int& width) {
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

  /**
   * reshape output image sizes
   */
  virtual void reshapeOutput(size_t height, size_t width) {
    output_.setFrameHeight(height);
    output_.setFrameWidth(width);
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      outputOtherDevice_[i].setFrameHeight(height);
      outputOtherDevice_[i].setFrameWidth(width);
    }
  }

  /**
   * reset MKLDNNMatrix from Matrix and internal primitive desc.
   * reset nullptr if matrix or primitive desc is empty
   */
  void resetWithMatrix(MKLDNNMatrixPtr& dnn,
                       const MatrixPtr& mat,
                       mkldnn::memory::primitive_desc pd) {
    dnn = nullptr;
    if (mat == nullptr) {
      return;
    }
    dnn = MKLDNNMatrix::create(pd, mat);
  }

  /**
   * reset input value from input MKLDNNMatrix and internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetInValue(
      MKLDNNMatrixPtr& in,
      const std::shared_ptr<mkldnn::memory::primitive_desc>& intPD = nullptr) {
    cvtInVal_ = nullptr;
    extInVal_ = nullptr;
    in = nullptr;
    CHECK_GT(bs_ * ic_ * ih_ * iw_, 0);
    auto extPD = MKLDNNMatrix::createPrimitiveDesc(
        {bs_, ic_, ih_, iw_}, mkldnn::memory::format::nchw, engine_);
    const MatrixPtr& inMat = inputLayers_[0]->getOutputValue();
    in = std::dynamic_pointer_cast<MKLDNNMatrix>(inMat);
    CHECK_EQ(inputIsOnlyMKLDNN(), in != nullptr);
    if (in == nullptr || in->getFormat() == mkldnn::memory::format::nc) {
      in = MKLDNNMatrix::create(extPD, inMat);
    }
    extInVal_ = isPaddleFormat(in->getFormat()) ? in : nullptr;
    if (in->getFormat() == mkldnn::memory::format::nc) {
      CHECK(ih_ == 1 && iw_ == 1);
    }
    if (nullptr == intPD || in->getPrimitiveDesc() == *intPD) {
      return;
    }
    // need create reorder
    in = MKLDNNMatrix::create(*intPD);
    extInVal_ = extInVal_ ? extInVal_ : MKLDNNMatrix::create(extPD, inMat);
    cvtInVal_ = MKLDNNMatrix::createReorder(extInVal_, in);
    CHECK(cvtInVal_) << "should not be emptry";
  }

  /**
   * reset output value from internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetOutValue(MKLDNNMatrixPtr& out,
                     mkldnn::memory::primitive_desc intPD) {
    cvtOutVal_ = nullptr;
    out = MKLDNNMatrix::create(intPD, output_.value);
    extOutVal_ = out;
    if (outputIsOnlyMKLDNN() || isPaddleFormat(extOutVal_->getFormat())) {
      return;
    }
    // need create reorder
    CHECK_GT(bs_ * oc_ * oh_ * ow_, 0);
    extOutVal_ = MKLDNNMatrix::create(mkldnn::memory::dims{bs_, oc_, oh_, ow_},
                                      mkldnn::memory::format::nchw,
                                      engine_,
                                      output_.value);
    out = MKLDNNMatrix::create(intPD);
    cvtOutVal_ = MKLDNNMatrix::createReorder(out, extOutVal_);
    CHECK(cvtOutVal_) << "should not be empty";
  }

  /**
   * reset input grad from internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetInGrad(MKLDNNMatrixPtr& in, mkldnn::memory::primitive_desc intPD) {
    cvtInGrad_ = nullptr;
    extInGrad_ = nullptr;
    in = nullptr;
    LayerPtr& input = inputLayers_[0];
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
    CHECK(inVal_ != nullptr && inVal_->getPrimitiveDesc() == intPD)
        << "should have internal input value and primitive desc must equal";
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
    CHECK(inVal_ != nullptr && inVal_->getPrimitiveDesc() == intPD)
        << "should have internal input value and primitive desc must equal";
    in = MKLDNNMatrix::create(intPD);
    cvtInGrad_ = MKLDNNMatrix::createReorder(in, extInGrad_);
    CHECK(cvtInGrad_);
  }

  /**
   * reset output grad from internal primitive desc.
   * merge grad if necessary.
   * reset both internal and external buffer and create reorder if necessary.
   * note: about merge grad, when this layer has serval outputs,
   *       it could not be mixed with cpu device,
   *       since it can not get memory desc from cpu device.
   */
  void resetOutGrad(MKLDNNMatrixPtr& out,
                    mkldnn::memory::primitive_desc intPD) {
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
    CHECK(outVal_ != nullptr && outVal_->getPrimitiveDesc() == intPD)
        << "should have internal output value and primitive desc must equal";
    out = MKLDNNMatrix::create(intPD);
    cvtOutGrad_ = MKLDNNMatrix::createReorder(extOutGrad_, out);
    CHECK(cvtOutGrad_);
  }

  /**
   * reset the merge grad primitive if necessary.
   * note: do not support the grads are mixed with cpu device,
   *       since it can not get memory desc from cpu device.
   */
  virtual void resetMergeGrad(MKLDNNMatrixPtr& out) {
    mergeGrad_ = nullptr;
    pipelineMergeGrad_.clear();
    if (outputMap_.size() <= 1 || !outputIsOnlyMKLDNN()) {
      // do not merge when output is not all MKLDNN or only one output
      return;
    }
    CHECK(out) << "should have reset internal ouput grad";
    std::vector<double> scales(outputMap_.size(), 1.0);
    std::vector<mkldnn::memory::primitive_desc> srcPDs;
    std::vector<mkldnn::primitive::at> srcs;
    for (auto it = outputMap_.begin(); it != outputMap_.end(); ++it) {
      MKLDNNMatrixPtr src =
          std::dynamic_pointer_cast<MKLDNNMatrix>(it->second->grad);
      VLOG(MKLDNN_BASE) << getName() << " has output grad " << it->first;
      CHECK(src) << "should be MKLDNNMatrix";
      auto srcDims = src->getDims();
      auto dstDims = out->getDims();
      CHECK_EQ(srcDims.size(), dstDims.size());
      for (size_t i = 0; i < srcDims.size(); ++i) {
        CHECK_EQ(srcDims[i], dstDims[i]);
      }
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

    auto sumPD = mkldnn::sum::primitive_desc(
        tmpOutGrad_->getMemoryDesc(), scales, srcPDs);
    mergeGrad_.reset(new mkldnn::sum(sumPD, srcs, *tmpOutGrad_));
    pipelineMergeGrad_.insert(pipelineMergeGrad_.begin(), *mergeGrad_);
  }

  /**
   * print info about sizes
   */
  virtual void printSizeInfo() {
    VLOG(MKLDNN_SIZES) << getName() << ": bs: " << bs_ << ", ic: " << ic_
                       << ", ih: " << ih_ << ", iw: " << iw_ << ", oc: " << oc_
                       << ", oh: " << oh_ << ", ow: " << ow_;
  }

  /**
   * print the mkldnn memory format of value
   */
  virtual void printValueFormat() {
    if (extInVal_) {
      VLOG(MKLDNN_FMTS) << extInVal_->getFormat() << " >>> ";
    }
    if (inVal_) {
      VLOG(MKLDNN_FMTS) << inVal_->getFormat() << " >>>";
    }
    if (outVal_) {
      VLOG(MKLDNN_FMTS) << outVal_->getFormat() << " >>> ";
    }
    if (extOutVal_) {
      VLOG(MKLDNN_FMTS) << extOutVal_->getFormat();
    }
    if (wgtVal_) {
      VLOG(MKLDNN_FMTS) << "Weight value format: " << wgtVal_->getFormat();
    }
    if (biasVal_) {
      VLOG(MKLDNN_FMTS) << "Bias value format: " << biasVal_->getFormat();
    }
  }

  /**
   * print the mkldnn memory format of grad
   */
  virtual void printGradFormat() {
    if (extInGrad_) {
      VLOG(MKLDNN_FMTS) << extInGrad_->getFormat() << " <<< ";
    }
    if (inGrad_) {
      VLOG(MKLDNN_FMTS) << inGrad_->getFormat() << " <<<";
    }
    if (outGrad_) {
      VLOG(MKLDNN_FMTS) << outGrad_->getFormat() << " <<< ";
    }
    if (extOutGrad_) {
      VLOG(MKLDNN_FMTS) << extOutGrad_->getFormat();
    }
    if (wgtGrad_) {
      VLOG(MKLDNN_FMTS) << "Weight grad format: " << wgtGrad_->getFormat();
    }
    if (biasGrad_) {
      VLOG(MKLDNN_FMTS) << "Bias grad format: " << biasGrad_->getFormat();
    }
  }

protected:
  /**
   * If input only has MKLDNN device.
   * Otherwise, only support the previous layer using CPU device.
   */
  bool inputIsOnlyMKLDNN(int index = 0) {
    int prevDevice = getPrev(index)->getDeviceId();
    if (prevDevice == MKLDNN_DEVICE) {
      return true;
    } else {
      // do not support GPU yet
      CHECK_EQ(prevDevice, CPU_DEVICE) << "Only support CPU yet";
      return false;
    }
  }

  /**
   * If output only has MKLDNN device.
   * Otherwise, other devices should only using CPU device.
   */
  bool outputIsOnlyMKLDNN() {
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      CHECK_EQ(outputOtherDevice_[i].deviceId, CPU_DEVICE)
          << "Only support other device is CPU yet";
    }
    outputOnlyMKLDNN_ = outputOtherDevice_.size() == 0;
    return outputOnlyMKLDNN_;
  }

  /**
   * Set deviceId of this layer.
   */
  void setDevice(int id) { deviceId_ = id; }

private:
  /**
   * check the format is nchw or nc,
   * which is supported by Paddle default memory layout
   */
  bool isPaddleFormat(mkldnn::memory::format fmt) {
    if (fmt == mkldnn::memory::format::nchw ||
        fmt == mkldnn::memory::format::nc) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * clear all grad
   */
  void clearGrads() {
    output_.grad->zeroMem();
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      outputOtherDevice_[i].grad->zeroMem();
    }
  }

  /**
   * Set deviceId of the params used in this layer.
   */
  void setParamsDevice(int id, const ParameterMap& parameterMap) {
    for (auto& inputConfig : config_.inputs()) {
      if (inputConfig.has_input_parameter_name()) {
        ParameterPtr parameter;
        std::string name = inputConfig.input_parameter_name();
        CHECK(mapGet(name, parameterMap, &parameter))
            << "Cannot find input parameter " << name << " for layer "
            << getName();
        parameter->setDevice(id);
      }
    }
    if (config_.has_bias_parameter_name()) {
      ParameterPtr parameter;
      std::string name = config_.bias_parameter_name();
      CHECK(mapGet(name, parameterMap, &parameter))
          << "Cannot find bias parameter " << name << " for layer "
          << getName();
      parameter->setDevice(id);
    }
  }

  /**
   * Set output map of prev layers.
   */
  void setOutputMap() {
    outputMap_.clear();
    for (size_t i = 0; i < inputLayers_.size(); ++i) {
      inputLayers_[i]->setOutput(getName(), &tmpInArg_);
    }
  }

  /**
   * if have cpu device, share value and grad data with output_
   */
  void shareCPUDevice() {
    if (outputIsOnlyMKLDNN()) {
      return;
    }
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      outputOtherDevice_[i].value = output_.value;
      outputOtherDevice_[i].grad = output_.grad;
    }
  }

  /**
   * Check the cpu device number of outputOtherDevice_.
   * should have only one at most.
   */
  void checkCPUOutputsNumber(int max = 1) {
    int cnt = 0;
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      if (outputOtherDevice_[i].deviceId == CPU_DEVICE) {
        ++cnt;
      }
    }
    CHECK_LE(cnt, max) << "too much CPU devies";
  }

  /**
   * copy SeqInfo from input layer to this output and other output devices.
   * @note: do not use getInput(0) since it used this deviceId_,
   *        use "inputLayers_[0]->getOutput()" instead.
   */
  void copySeqInfoToOutputs() {
    if (inputLayers_.empty() || !needSequenceInfo_) {
      return;
    }
    const Argument& input = inputLayers_[0]->getOutput();
    output_.sequenceStartPositions = input.sequenceStartPositions;
    output_.subSequenceStartPositions = input.subSequenceStartPositions;
    output_.cpuSequenceDims = input.cpuSequenceDims;
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      outputOtherDevice_[i].sequenceStartPositions =
          output_.sequenceStartPositions;
      outputOtherDevice_[i].subSequenceStartPositions =
          output_.subSequenceStartPositions;
      outputOtherDevice_[i].cpuSequenceDims = output_.cpuSequenceDims;
    }
  }
};

}  // namespace paddle
