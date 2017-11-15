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
 * @brief A subclass of MKLDNNLayer Addto layer.
 *
 * The config file api is mkldnn_addto
 */
class MKLDNNAddtoLayer : public MKLDNNLayer {
protected:
  std::vector<MKLDNNMatrixPtr> inVals_;
  std::vector<MKLDNNMatrixPtr> inGrads_;

  // layer size == ic * ih * iw == oc * oh *ow, and can not be changed
  size_t layerSize_;

  std::unique_ptr<Weight> biases_;

  // buffers for adding bias
  std::vector<MKLDNNMatrixPtr> vals_;
  std::vector<MKLDNNMatrixPtr> grads_;
  // primitives for adding bias
  std::vector<std::shared_ptr<mkldnn::primitive>> fwdBias_;
  std::shared_ptr<mkldnn::primitive> bwdBias_;

public:
  explicit MKLDNNAddtoLayer(const LayerConfig& config) : MKLDNNLayer(config) {}

  ~MKLDNNAddtoLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void reshape(
      int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void updateWeights(const UpdateCallback& callback) override;

  void printValueFormat() override {
    for (size_t i = 0; i < inVals_.size(); ++i) {
      VLOG(MKLDNN_FMTS) << i << " input: " << inVals_[i]->getFormat() << " >>>";
    }
    if (outVal_) {
      VLOG(MKLDNN_FMTS) << outVal_->getFormat() << " >>> ";
    }
    if (extOutVal_) {
      VLOG(MKLDNN_FMTS) << extOutVal_->getFormat();
    }
  }

  void printGradFormat() override {
    if (extOutGrad_) {
      VLOG(MKLDNN_FMTS) << extOutGrad_->getFormat();
    }
    if (outGrad_) {
      VLOG(MKLDNN_FMTS) << outGrad_->getFormat() << " <<< ";
    }
    for (size_t i = 0; i < inGrads_.size(); ++i) {
      VLOG(MKLDNN_FMTS) << i << " input: " << inGrads_[i]->getFormat() << "<<<";
    }
  }

protected:
  /**
   * Forward functions: reset buffers(inputs, output, bias),
   *                    reset primitive descriptor,
   *                    reset pipeline.
   */
  void resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<mkldnn::sum::primitive_desc>& pd,
                  std::shared_ptr<mkldnn::sum::primitive_desc>& biasPD,
                  std::vector<MKLDNNMatrixPtr>& inputs,
                  MKLDNNMatrixPtr bias,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<mkldnn::sum::primitive_desc>& pd,
                        std::shared_ptr<mkldnn::sum::primitive_desc>& biasPD,
                        std::vector<MKLDNNMatrixPtr>& inputs,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out);

  /**
   * Backward functions: reset buffers(inputs, output, bias)
   */
  void resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);

  /**
   * prepare for bias
   */
  void prepareBias(MKLDNNMatrixPtr& bias,
                   const MatrixPtr& biasMat,
                   const MKLDNNMatrixPtr& out,
                   std::vector<MKLDNNMatrixPtr>& outs);
};

}  // namespace paddle
