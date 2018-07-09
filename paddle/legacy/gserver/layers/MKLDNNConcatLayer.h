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

/**
 * @brief A subclass of MKLDNNLayer Concatenate layer.
 *
 * The config file api is mkldnn_concat
 */
class MKLDNNConcatLayer : public MKLDNNLayer {
 protected:
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
      int& bs, int& ic, int& ih, int& iw, int& oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                std::vector<MKLDNNMatrixPtr>& inputs,
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

  size_t keepCondition() {
    // reset when the total element size of all inputs changed
    size_t totalSize = inputLayers_[0]->getOutputValue()->getElementCnt();
    for (size_t i = 1; i < inputLayers_.size(); ++i) {
      totalSize += inputLayers_[i]->getOutputValue()->getElementCnt();
    }
    return totalSize;
  }

 protected:
  void resetFwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<mkldnn::concat::primitive_desc>& pd,
                  std::vector<MKLDNNMatrixPtr>& inputs,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<mkldnn::concat::primitive_desc>& pd,
                        std::vector<MKLDNNMatrixPtr>& inputs,
                        MKLDNNMatrixPtr& out);
  void resetBwdBuffers(std::vector<MKLDNNMatrixPtr>& inputs,
                       MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::vector<std::shared_ptr<mkldnn::primitive>>& prims,
                        std::vector<MKLDNNMatrixPtr>& inputs,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
