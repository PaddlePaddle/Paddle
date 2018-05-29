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
typedef mkldnn::pooling_forward pool_fwd;
typedef mkldnn::pooling_backward pool_bwd;

/**
 * @brief A subclass of MKLDNNLayer pool layer.
 *
 * The config file api is mkldnn_pool
 */
class MKLDNNPoolLayer : public MKLDNNLayer {
 protected:
  // padding height and width
  int ph_, pw_;
  // stride height and width
  int sh_, sw_;
  // filter(kenerl) height and width
  int fh_, fw_;

  // pooling_avg or pooling_max
  mkldnn::algorithm poolAlgo_;

  // save forward primitive_desc, which can be used backward
  std::shared_ptr<pool_fwd::primitive_desc> fwdPD_;
  // according to https://github.com/01org/mkl-dnn/blob/master/tests/gtests/
  // test_pooling_forward.cpp, pool need workspace for backward
  std::shared_ptr<mkldnn::memory> workspace_;

 public:
  explicit MKLDNNPoolLayer(const LayerConfig& config) : MKLDNNLayer(config) {}

  ~MKLDNNPoolLayer() {}

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

  void printSizeInfo() override {
    MKLDNNLayer::printSizeInfo();
    VLOG(MKLDNN_SIZES) << getName() << ": fh: " << fh_ << ", fw: " << fw_
                       << ": ph: " << ph_ << ", pw: " << pw_ << ", sh: " << sh_
                       << ", sw: " << sw_;
  }

 protected:
  void resetFwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<pool_fwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr in,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<pool_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);
  void resetBwdBuffers(MKLDNNMatrixPtr& in, MKLDNNMatrixPtr& out);
  void resetBwdPD(std::shared_ptr<pool_bwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr& in,
                  MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<pool_bwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& out);

  /**
   * get padding_r according to
   * https://github.com/01org/mkl-dnn/blob/master/tests/gtests/
   * test_pooling_forward.cpp
   */
  mkldnn::memory::dims getPaddingR() const {
    mkldnn::memory::dims padR = {ph_, pw_};
    for (int i = 0; i < 2; ++i) {
      if ((ih_ + ph_ + padR[0] - fh_) / sh_ + 1 < oh_) {
        ++padR[0];
      }
      if ((iw_ + pw_ + padR[1] - fw_) / sw_ + 1 < ow_) {
        ++padR[1];
      }
    }
    return padR;
  }
};

}  // namespace paddle
