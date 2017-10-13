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

  // mkldnn engine, stream and primivtives
  mkldnn::engine engine_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwdWgt_;
  std::shared_ptr<mkldnn::primitive> bwdData_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

  // MKLDNNMatrixPtr with internal format
  MKLDNNMatrixPtr inVal_;
  MKLDNNMatrixPtr inGrad_;
  MKLDNNMatrixPtr outVal_;
  MKLDNNMatrixPtr outGrad_;
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
      size_t elemenCnt = inputLayers_[0]->getOutput().value->getElementCnt();
      if (inputElemenCnt_ != elemenCnt) {
        VLOG(MKLDNN_BASE) << getName() << " reset mkldnn forward";
        // reset when input total sizes changed, not only the batchsize
        inputElemenCnt_ = elemenCnt;
        pipelineFwd_.clear();
        reshape(bs_, ic_, ih_, iw_, oc_, oh_, ow_);
        resetFwd(pipelineFwd_, inVal_, wgtVal_, biasVal_, outVal_);
        convertWeightsFromPaddle();
        needResetBwd_ = true;
      }

      if (inputLayers_[0]->getType() == "data") {
        updateInputData();
      }

      stream_->submit(pipelineFwd_);
    }

    /* activation */ {
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
   * reset the mkldnn forward primitve and memory
   * only would be called when input size changes
   */
  virtual void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out) = 0;

  /**
   * reset the mkldnn backward primitve and memory for mkldnn fc
   * only would be called when needed
   */
  virtual void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out) = 0;

  /**
   * Update input value data when input layer is "data" type.
   * Since the input value data address might be changed.
   */
  virtual void updateInputData() {}

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
   * reset the output grad matrix from primitive desc.
   * and reset the merge grad primitive if needed.
   * note: when this layer have serval output,
   *       do not support mixing with cpu device,
   *       because can not get memory desc from cpu device.
   */
  virtual void resetOutGrad(MKLDNNMatrixPtr& out,
                            mkldnn::memory::primitive_desc pd) {
    CHECK(outputIsOnlyMKLDNN()) << "do not support mixed with other device yet";
    mergeGrad_ = nullptr;
    pipelineMergeGrad_.clear();
    out = MKLDNNMatrix::create(output_.grad, pd);
    if (outputMap_.size() <= 1) {
      return;
    }
    std::vector<double> scales;
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
      scales.push_back(1.0);
    }

    // TODO(TJ): remove me when mkldnn sum support different formats
    for (size_t i = 1; i < srcPDs.size(); ++i) {
      CHECK(srcPDs[0] == srcPDs[i]);
    }
    tmpOutGrad_ = nullptr;
    tmpCvt_ = nullptr;
    if (out->getPrimitiveDesc() != srcPDs[0]) {
      tmpOutGrad_ = MKLDNNMatrix::create(nullptr, srcPDs[0]);
      tmpCvt_ = MKLDNNMatrix::createReorder(tmpOutGrad_, out);
      CHECK(tmpCvt_);
      pipelineMergeGrad_.push_back(*tmpCvt_);
    } else {
      tmpOutGrad_ = out;
    }

    auto sumPD = mkldnn::sum::primitive_desc(
        tmpOutGrad_->getMemoryDesc(), scales, srcPDs);
    mergeGrad_.reset(new mkldnn::sum(sumPD, srcs, *tmpOutGrad_));
    pipelineMergeGrad_.insert(pipelineMergeGrad_.begin(), *mergeGrad_);
  }

  /**
   * reset input grad from primitive desc.
   * this function is avaiable for input is only mkldnn
   * or input do not care cpu device
   */
  virtual void resetInGrad(MKLDNNMatrixPtr& in,
                           mkldnn::memory::primitive_desc pd) {
    LayerPtr& input = inputLayers_[0];
    const MatrixPtr& grad =
        input->getOutputMapSize() > 1 ? nullptr : input->getOutput().grad;
    in = MKLDNNMatrix::create(grad, pd);
    Argument& arg = input->getOutput(this->getName());
    arg.grad = std::dynamic_pointer_cast<Matrix>(in);
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
   * Print the mkldnn memory format flow of value
   */
  virtual void printValueFormatFlow() {
    if (inVal_ && outVal_) {
      VLOG(MKLDNN_FMTS) << inVal_->getFormat() << " >>> "
                        << outVal_->getFormat();
    }
  }

  /**
   * Print the mkldnn memory format flow of grad
   */
  virtual void printGradFormatFlow() {
    if (inGrad_ && outGrad_) {
      VLOG(MKLDNN_FMTS) << inGrad_->getFormat() << " <<< "
                        << outGrad_->getFormat();
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
    return outputOtherDevice_.size() == 0;
  }

  /**
   * Set deviceId of this layer.
   */
  void setDevice(int id) { deviceId_ = id; }

private:
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
