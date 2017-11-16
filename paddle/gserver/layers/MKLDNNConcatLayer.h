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
 * @brief A subclass of MKLDNNLayer Concatenate layer.
 *
 * The config file api is mkldnn_concat
 */
class MKLDNNConcatLayer : public MKLDNNLayer {
protected:
  std::vector<MKLDNNMatrixPtr> inVals_;
  std::vector<MKLDNNMatrixPtr> inGrads_;
  std::vector<std::shared_ptr<mkldnn::primitive>> bwds_;
  // input channel numbers
  std::vector<int> channels_;

  // concat_dimension in MKLDNN
  // if axis_ == 0, concat batchsize
  // if axis_ == 1, concat channel (default)
  int axis_;

public:
  explicit MKLDNNConcatLayer(const LayerConfig& config)
      : MKLDNNLayer(config), axis_(1) {}

  ~MKLDNNConcatLayer() {}

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

  void printSizeInfo() override {
    CHECK_EQ(channels_.size(), inputLayers_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
      VLOG(MKLDNN_SIZES) << "Input " << i << ", " << inputLayers_[i]->getName()
                         << ": " << bs_ << ", " << channels_[i] << ", " << ih_
                         << ", " << iw_;
    }
    VLOG(MKLDNN_SIZES) << "Output: " << bs_ << ", " << oc_ << ", " << oh_
                       << ", " << ow_;
  }

  void printValueFormat() override {
    for (size_t i = 0; i < inVals_.size(); ++i) {
      VLOG(MKLDNN_FMTS) << "Input " << i << inputLayers_[i]->getName() << ": "
                        << inVals_[i]->getFormat() << " >>>";
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
      VLOG(MKLDNN_FMTS) << "Input " << i << inputLayers_[i]->getName() << ": "
                        << inGrads_[i]->getFormat() << "<<<";
    }
  }

protected:
  /**
   * Forward functions: reset buffers(inputs, output, bias),
   *                    reset primitive descriptor,
   *                    reset pipeline.
   */
  void resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<mkldnn::concat::primitive_desc>& pd,
                  std::vector<MKLDNNMatrixPtr>& inputs,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<mkldnn::concat::primitive_desc>& pd,
                        std::vector<MKLDNNMatrixPtr>& inputs,
                        MKLDNNMatrixPtr& out);

  /**
   * Backward functions: reset buffers(inputs, output, bias)
   *                     reset primitives and pipeline
   */
  void resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::vector<std::shared_ptr<mkldnn::primitive>>& prims,
                        std::vector<MKLDNNMatrixPtr>& inputs,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
