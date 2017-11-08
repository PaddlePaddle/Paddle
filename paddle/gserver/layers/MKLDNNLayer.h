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

  /* Value and grad are seperated as internal and external buffers.
   * Each MKLDNNLayer must init or reset internal buffer at least,
   * and the external buffer format is always nchw of nc(when h==w==1),
   * which is the same format as paddle.
   * The output_.value and output_.grad always save the external data,
   * when mixed with cpu device.
   * When all layers are mkldnn layers, they could save internal data.
   */
  // below MKLDNNMatrix buffers are all internal buffers
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

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  virtual void forward(PassType passType);
  virtual void backward(const UpdateCallback& callback);

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
  void reshapeInput(int& batchsize, int& height, int& width);

  /**
   * reshape output image sizes
   */
  void reshapeOutput(size_t height, size_t width);

  /**
   * reset MKLDNNMatrix from Matrix and internal primitive desc.
   * reset nullptr if matrix or primitive desc is empty
   */
  void resetWithMatrix(MKLDNNMatrixPtr& dnn,
                       const MatrixPtr& mat,
                       mkldnn::memory::primitive_desc pd);

  /**
   * reset input value from input MKLDNNMatrix and internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetInValue(
      MKLDNNMatrixPtr& in,
      const std::shared_ptr<mkldnn::memory::primitive_desc>& intPD = nullptr,
      size_t inputIdx = 0);

  /**
   * reset output value from internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetOutValue(MKLDNNMatrixPtr& out,
                     mkldnn::memory::primitive_desc intPD);

  /**
   * reset input grad from internal primitive desc.
   * reset both internal and external buffer and create reorder if necessary.
   */
  void resetInGrad(MKLDNNMatrixPtr& in,
                   mkldnn::memory::primitive_desc intPD,
                   size_t inputIdx = 0);

  /**
   * reset output grad from internal primitive desc.
   * merge grad if necessary.
   * reset both internal and external buffer and create reorder if necessary.
   * note: about merge grad, when this layer has several outputs,
   *       it could not be mixed with cpu device,
   *       since it can not get memory desc from cpu device.
   */
  void resetOutGrad(MKLDNNMatrixPtr& out, mkldnn::memory::primitive_desc intPD);

  /**
   * reset the merge grad primitive if necessary.
   * note: do not support the grads mixed with cpu device,
   *       since it can not get memory desc from cpu device.
   */
  void resetMergeGrad(MKLDNNMatrixPtr& out);

protected:
  /**
   * Set deviceId of this layer.
   */
  void setDevice(int id) { deviceId_ = id; }

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
   * If input only has MKLDNN device.
   * Otherwise, only support the previous layer using CPU device.
   */
  bool inputIsOnlyMKLDNN(int index = 0) {
    int prevDevice = getPrev(index)->getDeviceId();
    if (prevDevice == MKLDNN_DEVICE) {
      return true;
    } else {
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
    if (extOutGrad_) {
      VLOG(MKLDNN_FMTS) << extOutGrad_->getFormat();
    }
    if (outGrad_) {
      VLOG(MKLDNN_FMTS) << outGrad_->getFormat() << " <<< ";
    }
    if (inGrad_) {
      VLOG(MKLDNN_FMTS) << inGrad_->getFormat() << " <<<";
    }
    if (extInGrad_) {
      VLOG(MKLDNN_FMTS) << extInGrad_->getFormat() << " <<< ";
    }
    if (wgtGrad_) {
      VLOG(MKLDNN_FMTS) << "Weight grad format: " << wgtGrad_->getFormat();
    }
    if (biasGrad_) {
      VLOG(MKLDNN_FMTS) << "Bias grad format: " << biasGrad_->getFormat();
    }
  }

private:
  /**
   * clear all grad
   */
  void clearGrads() {
    if (output_.grad) {
      output_.grad->zeroMem();
    }
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      if (outputOtherDevice_[i].grad) {
        outputOtherDevice_[i].grad->zeroMem();
      }
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
