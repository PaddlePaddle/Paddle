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

#pragma once

#include "MKLDNNLayer.h"
#include "mkldnn.hpp"

namespace paddle {
typedef mkldnn::convolution_forward conv_fwd;
typedef mkldnn::convolution_backward_weights conv_bwdWgt;
typedef mkldnn::convolution_backward_data conv_bwdData;

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

  // in resetBwdData, the format of wgtValBwdData_ is different with wgtVal_
  MKLDNNMatrixPtr wgtValBwdData_;
  // convert handle from wgtVal_ to wgtValBwdData_
  std::shared_ptr<mkldnn::reorder> cvtWgtVal_;

  // save forward primitive_desc, which can be used backward
  std::shared_ptr<conv_fwd::primitive_desc> fwdPD_;

  // whether the weight has been init
  bool hasInitedWgt_;

  // true by default, which impact the calculation of output image size.
  // details can refer to mathUtil.h
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

  void reshape(
      int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
                MKLDNNMatrixPtr& out) override;

  void updateWeights(const UpdateCallback& callback) override;

  void convertWeightsFromPaddle() override;

  void convertWeightsToPaddle() override;

  void printSizeInfo() override {
    MKLDNNLayer::printSizeInfo();
    VLOG(MKLDNN_SIZES) << getName() << ": fh: " << fh_ << ", fw: " << fw_
                       << ", ph: " << ph_ << ", pw: " << pw_ << ", sh: " << sh_
                       << ", sw: " << sw_ << ", dh: " << dh_ << ", dw: " << dw_;
  }

 protected:
  /**
   * load the dims settings of this conv
   */
  void loadConvSettings(mkldnn::memory::dims& wgt,
                        mkldnn::memory::dims& bias,
                        mkldnn::memory::dims& stride,
                        mkldnn::memory::dims& dilation,
                        mkldnn::memory::dims& padL,
                        mkldnn::memory::dims& padR);

  void resetFwdPD(std::shared_ptr<conv_fwd::primitive_desc>& pd);
  void resetFwdBuffers(std::shared_ptr<conv_fwd::primitive_desc>& pd,
                       MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<conv_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out);
  void resetBwdWgtPD(std::shared_ptr<conv_bwdWgt::primitive_desc>& pd);
  void resetBwdDataPD(std::shared_ptr<conv_bwdData::primitive_desc>& pd);
  void resetBwdBuffers(std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
                       std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
                       MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<conv_bwdWgt::primitive_desc>& wgtPD,
                        std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out);

  /**
   * reset MKLDNNMatrix of weight value for backward data
   * since the primitive_desc would be different with wgtVal_
   */
  void resetWgtValBwdData(std::shared_ptr<conv_bwdData::primitive_desc>& dataPD,
                          MKLDNNMatrixPtr& wgt);

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
