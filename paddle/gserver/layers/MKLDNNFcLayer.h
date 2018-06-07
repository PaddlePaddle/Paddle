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
typedef mkldnn::inner_product_forward fc_fwd;
typedef mkldnn::inner_product_backward_weights fc_bwdWgt;
typedef mkldnn::inner_product_backward_data fc_bwdData;

/**
 * @brief A subclass of MKLDNNLayer fc layer.
 *
 * The config file api is mkldnn_fc
 */
class MKLDNNFcLayer : public MKLDNNLayer {
 protected:
  // input layer size, can not be change after init
  size_t iLayerSize_;  // == ic * ih * iw

  // if has already init the weight
  bool hasInitedWgt_;

  // save forward primitive_desc, which can be used backward
  std::shared_ptr<fc_fwd::primitive_desc> fwdPD_;

  // fc weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

 public:
  explicit MKLDNNFcLayer(const LayerConfig& config)
      : MKLDNNLayer(config), hasInitedWgt_(false) {}

  ~MKLDNNFcLayer() {}

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

 protected:
  void resetFwdBuffers(MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<fc_fwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr in,
                  MKLDNNMatrixPtr wgt,
                  MKLDNNMatrixPtr bias,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<fc_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out);
  void resetBwdBuffers(MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& bias,
                       MKLDNNMatrixPtr& out);
  void resetBwdWgtPD(std::shared_ptr<fc_bwdWgt::primitive_desc>& pd,
                     MKLDNNMatrixPtr& wgt,
                     MKLDNNMatrixPtr& bias,
                     MKLDNNMatrixPtr& out);
  void resetBwdDataPD(std::shared_ptr<fc_bwdData::primitive_desc>& pd,
                      MKLDNNMatrixPtr& in,
                      MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<fc_bwdWgt::primitive_desc>& bwdWgtPD,
                        std::shared_ptr<fc_bwdData::primitive_desc>& bwdDataPD,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& bias,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
