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

  // MKLDNNMatrixPtr with user format
  // in case for conversion with CPU device or MKLDNN device with other format
  MKLDNNMatrixPtr userInVal_;
  MKLDNNMatrixPtr userOutVal_;
  MKLDNNMatrixPtr userInGrad_;
  MKLDNNMatrixPtr userOutGrad_;
  std::shared_ptr<mkldnn::reorder> cvtInVal_;
  std::shared_ptr<mkldnn::reorder> cvtOutVal_;
  std::shared_ptr<mkldnn::reorder> cvtInGrad_;
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
  // internal matrix for input value and output value
  MatrixPtr internalInVal_;
  MatrixPtr internalOutVal_;

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

  void printSizeInfo();

protected:
  void printValueFormatFlow() override {
    if (userInVal_) {
      VLOG(MKLDNN_FMTS) << userInVal_->getFormat() << " >>>";
    }
    MKLDNNLayer::printValueFormatFlow();
    if (userOutVal_) {
      VLOG(MKLDNN_FMTS) << " >>> " << userOutVal_->getFormat();
    }
  }
  void printGradFormatFlow() override {
    if (userInGrad_) {
      VLOG(MKLDNN_FMTS) << userInGrad_->getFormat() << " <<<";
    }
    MKLDNNLayer::printGradFormatFlow();
    if (userOutGrad_) {
      VLOG(MKLDNN_FMTS) << " <<< " << userOutGrad_->getFormat();
    }
  }

private:
  /**
   * get padding_r according to
   * https://github.com/01org/mkl-dnn/blob/master/tests/gtests/
   * test_convolution_forward_common.hpp
   * @note: mkldnn dilation start from 0 while paddle start from 1
   */
  mkldnn::memory::dims getPaddingR() const;
};

}  // namespace paddle
