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
typedef mkldnn::batch_normalization_forward bn_fwd;
typedef mkldnn::batch_normalization_backward bn_bwd;

/**
 * @brief A subclass of MKLDNNLayer BatchNorm layer.
 *
 * The config file api is mkldnn_batch_norm
 */
class MKLDNNBatchNormLayer : public MKLDNNLayer {
 protected:
  // save forward primitive_desc, which can be used backward
  std::shared_ptr<bn_fwd::primitive_desc> fwdPD_;

  // Epsilon value used in the batch normalization formula.
  real epsilon_;

  // weight and bias in paddle
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;
  // mkldnn use a large buffer store both scale and shift
  // which are weight and bias in paddle corresponding.
  MatrixPtr valueScaleShift_;
  MatrixPtr gradScaleShift_;
  // Moving average of mean.
  std::unique_ptr<Weight> movingMean_;
  // Moving average of variance.
  std::unique_ptr<Weight> movingVar_;

  // if useGlobalStats_ is true, will use the loaded mean and variance.
  // otherwise, calculate mean and variance in every mini-batch.
  bool useGlobalStats_;
  // used in MKLDNN primitive desc
  unsigned flags_;
  // use to compute moving mean and variance.
  real movingAvgFraction_;
  // whether the weight has been init
  bool hasInitedWgt_;

  // local mean and variance
  // when useGlobalStats_ they are loaded from moving mean and variance
  // when do not useGlobalStats_ they are calculated from this mini-batch
  MKLDNNMatrixPtr mean_;
  MKLDNNMatrixPtr var_;

 public:
  explicit MKLDNNBatchNormLayer(const LayerConfig& config)
      : MKLDNNLayer(config), useGlobalStats_(true), hasInitedWgt_(false) {}

  ~MKLDNNBatchNormLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

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

 protected:
  void initWeight();
  /**
   * cal moving mean and variance.
   * moving = moving * AvgFraction + local * (1 - AvgFraction)
   */
  void calMovingMeanAndVar();

  void resetFwdBuffers(MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& out);
  void resetFwdPD(std::shared_ptr<bn_fwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr in,
                  MKLDNNMatrixPtr wgt,
                  MKLDNNMatrixPtr out);
  void resetFwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<bn_fwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& out);
  void resetBwdBuffers(MKLDNNMatrixPtr& in,
                       MKLDNNMatrixPtr& wgt,
                       MKLDNNMatrixPtr& out);
  void resetBwdPD(std::shared_ptr<bn_bwd::primitive_desc>& pd,
                  MKLDNNMatrixPtr& in,
                  MKLDNNMatrixPtr& wgt,
                  MKLDNNMatrixPtr& out);
  void resetBwdPipeline(std::vector<mkldnn::primitive>& pipeline,
                        std::shared_ptr<bn_bwd::primitive_desc>& pd,
                        MKLDNNMatrixPtr& in,
                        MKLDNNMatrixPtr& wgt,
                        MKLDNNMatrixPtr& out);
};

}  // namespace paddle
