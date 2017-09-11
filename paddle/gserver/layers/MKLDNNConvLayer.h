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

#include "MKLDNNLayer.h"
#include "mkldnn.hpp"

namespace paddle {

/**
 * @brief A subclass of MKLDNNLayer conv layer.
 *
 * The config file api is mkldnn_conv
 */
class MKLDNNConvLayer : public MKLDNNLayer {
protected:
  // padding height and width
  int ph_, pw_;
  // stride height and width
  int sh_, sw_;
  // dilation height and width
  int dh_, dw_;
  // filter(kenerl) height and width
  int fh_, fw_;
  // group number
  int gp_;

  // in backward data the format is different with wgtVal_
  MKLDNNMatrixPtr wgtValBwdData_;
  std::shared_ptr<mkldnn::reorder> cvtWgtVal_;

  // save forward primitive_desc use for backward
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwdPD_;

  // MKLDNNMatrixPtr with cpu device for conversion between MKLDNN device
  MKLDNNMatrixPtr cpuInVal_;
  MKLDNNMatrixPtr cpuInGrad_;
  MKLDNNMatrixPtr cpuOutVal_;
  MKLDNNMatrixPtr cpuOutGrad_;
  std::shared_ptr<mkldnn::reorder> cvtInVal_;
  std::shared_ptr<mkldnn::reorder> cvtInGrad_;
  std::shared_ptr<mkldnn::reorder> cvtOutVal_;
  std::shared_ptr<mkldnn::reorder> cvtOutGrad_;

  // if has already init the weight
  bool hasInitedWgt_;

  // True by default. This impact the calculation of output size.
  // For example:
  // - input(+padding): 0123456789
  // - imageSize(+padding) = 10;
  // - filterSize = 3;
  // - stride = 2;
  // - caffeMode_ is true:
  // - output: (012), (234), (456), (678)
  // - outputSize = 4;
  // - caffeMode_ is false:
  // - output: (012), (234), (456), (678), (9)
  // - outputSize = 5;
  bool caffeMode_;

  // weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

public:
  explicit MKLDNNConvLayer(const LayerConfig& config)
      : MKLDNNLayer(config), hasInitedWgt_(false), caffeMode_(true) {}

  ~MKLDNNConvLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void reshape() override;

  void resetFwd() override;

  void resetBwd() override;

  void updateInputData() override;

  void updateWeights(const UpdateCallback& callback) override;

  void convertWeightsFromPaddle() override;

  void convertWeightsToPaddle() override;

protected:
  void printSizeInfo() override {
    MKLDNNLayer::printSizeInfo();
    VLOG(MKLDNN_SIZES) << getName() << ": fh: " << fh_ << ", fw: " << fw_
                       << ": ph: " << ph_ << ", pw: " << pw_ << ", sh: " << sh_
                       << ", sw: " << sw_ << ", dh: " << dh_ << ", dw: " << dw_;
  }

  void printValueFormatFlow() override {
    if (cpuInVal_) {
      VLOG(MKLDNN_FMTS) << cpuInVal_->getFormat() << " >>>";
    }
    MKLDNNLayer::printValueFormatFlow();
    if (cpuOutVal_) {
      VLOG(MKLDNN_FMTS) << " >>> " << cpuOutVal_->getFormat();
    }
  }
  void printGradFormatFlow() override {
    if (cpuInGrad_) {
      VLOG(MKLDNN_FMTS) << cpuInGrad_->getFormat() << " <<<";
    }
    MKLDNNLayer::printGradFormatFlow();
    if (cpuOutGrad_) {
      VLOG(MKLDNN_FMTS) << " <<< " << cpuOutGrad_->getFormat();
    }
  }

  /**
   * get padding_r according to
   * https://github.com/01org/mkl-dnn/blob/master/tests/gtests/
   * test_convolution_forward_common.hpp
   * @note: mkldnn dilation start from 0 while paddle start from 1
   */
  mkldnn::memory::dims getPaddingR() const {
    mkldnn::memory::dims padR = {ph_, pw_};
    for (int i = 0; i < 2; ++i) {
      if ((ih_ - ((fh_ - 1) * dh_ + 1) + ph_ + padR[0]) / sh_ + 1 != oh_) {
        ++padR[0];
      }
      if ((iw_ - ((fw_ - 1) * dw_ + 1) + pw_ + padR[1]) / sw_ + 1 != ow_) {
        ++padR[1];
      }
    }
    return padR;
  }
};

}  // namespace paddle
