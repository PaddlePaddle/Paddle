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

  // cpu output Argument index of outputOtherDevice_, set -1 if none
  // TODO(TJ): remove me if unused
  int cpuOutputIndex_;

  // MKLDNNMatrixPtr with internal format
  MKLDNNMatrixPtr inVal_;
  MKLDNNMatrixPtr inGrad_;
  MKLDNNMatrixPtr outVal_;
  MKLDNNMatrixPtr outGrad_;
  MKLDNNMatrixPtr wgtVal_;
  MKLDNNMatrixPtr wgtGrad_;
  MKLDNNMatrixPtr biasVal_;
  MKLDNNMatrixPtr biasGrad_;

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
    cpuOutputIndex_ = getCPUOutputIndex();

    stream_.reset(new MKLDNNStream());
    engine_ = CPUEngine::Instance().getEngine();
    return true;
  }

  void forward(PassType passType) override {
    passType_ = passType;

    {
      REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
      copySeqInfoToOutputs();
      CHECK(!inputLayers_.empty());
      size_t elemenCnt = inputLayers_[0]->getOutput().value->getElementCnt();
      if (inputElemenCnt_ != elemenCnt) {
        inputElemenCnt_ = elemenCnt;
        reshape();
        resetFwd();
        convertWeightsFromPaddle();
        needResetBwd_ = true;
      }

      updateInputData();
      stream_->submit(pipelineFwd_);
    }

    /* activation */ {
      REGISTER_TIMER_INFO("FwActTimer", getName().c_str());
      forwardActivation();
    }
  }

  void backward(const UpdateCallback& callback) override {
    /* Do derivation */ {
      REGISTER_TIMER_INFO("BpActTimer", getName().c_str());
      backwardActivation();
    }

    {
      REGISTER_TIMER_INFO("mkldnn_bwdTimer", getName().c_str());
      if (needResetBwd_) {
        resetBwd();
        needResetBwd_ = false;
      }

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
   */
  virtual void reshape() = 0;

  /**
   * reset the mkldnn forward primitve and memory
   * only would be called when input size changes
   */
  virtual void resetFwd() = 0;

  /**
   * reset the mkldnn backward primitve and memory for mkldnn fc
   * only would be called when needed
   */
  virtual void resetBwd() = 0;

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
  virtual void reshapeInput() {
    const Argument& input = inputLayers_[0]->getOutput();
    bs_ = input.getBatchSize();
    int height = input.getFrameHeight();
    int width = input.getFrameWidth();
    if (height != 0) {
      ih_ = height;
    }
    if (width != 0) {
      iw_ = width;
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
   * get cpu output Argument index of outputOtherDevice_.
   * return -1 if none.
   */
  int getCPUOutputIndex() {
    int index = -1;
    int cnt = 0;
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      if (outputOtherDevice_[i].deviceId == CPU_DEVICE) {
        ++cnt;
        index = i;
      }
    }
    CHECK_LE(cnt, 1) << "should not have more than one CPU devie";
    return index;
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
