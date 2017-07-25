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

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer fc layer.
 *
 * The config file api is mkldnn_fc
 */
class MkldnnFcLayer : public MkldnnLayer {
protected:
  // TODO(TJ): move to local var
  std::shared_ptr<mkldnn::inner_product_forward> fwd_;
  std::shared_ptr<mkldnn::inner_product_backward_data> bwdData_;
  std::shared_ptr<mkldnn::inner_product_backward_weights> bwdWgt_;

  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

  /// dim of input layer, it can not be changed even in reshape
  size_t dim_in_;
  /// dim(== output change) of output layer, it also can not be changed
  size_t dim_out_;

  // fc weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

  /// The layout of paddle weight is different with mkldnn
  /// 1. initial weight from paddle
  /// 2. inference with paddle format wgt if do not use mkldnn wgt as input
  MatrixPtr paddleWgt_;

  /// weight data and diff buffers
  MkldnnBufferPtr wgtData_;
  MkldnnBufferPtr wgtDiff_;
  /// weight data and diff buffers
  MkldnnBufferPtr biasData_;
  MkldnnBufferPtr biasDiff_;

  /// top diff for backward weight
  // it may have different format with topdiff backward data
  MkldnnBufferPtr topDiffBwdWgt_;

  // if image width and height !=0
  bool hasSpatial_;

  bool hasBias_;

  /// only init wgt from paddle once
  bool hasInitedWgt_;

public:
  explicit MkldnnFcLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      paddleWgt_(nullptr),
      wgtData_(nullptr),
      wgtDiff_(nullptr),
      biasData_(nullptr),
      biasDiff_(nullptr),
      topDiffBwdWgt_(nullptr),
      hasSpatial_(false),
      hasBias_(false),
      hasInitedWgt_(false)
    {}

  ~MkldnnFcLayer() {}

  /**
   * load the input dim and output dim from proto
   */
  void loadConfig();

  /**
   * init weight buffer in paddle
   */
  bool initWgt(const LayerMap& layerMap, const ParameterMap& parameterMap);

  /**
   * reshape the output matrix height and width
   * reshape the batchsize
   * reshape the input and output image size
   * reshape the output image size
   */
  void reshape();

  void resetFwd(PassType passType);

  void resetBwd();

  void submitFwd();

  void submitBwd(const UpdateCallback& callback);

  /**
   * Init mkldnn weight from paddle format only once
   * when training or scoring from paddle format
   * however when scoring with mkldnn wgt donot need initial from paddle format
   */
  void initWgtFromPaddle();

  /**
   * Convert the mkldnn weights to the paddle format
   * This functions could be called in gtest
   */
  void cvtWgtToPaddle() override {
    // TODO(TJ): enable me
  }

protected:
  // fc can only change the out mat height but not width(layersize)
  // since when input width changed, the wgt would need be changed too
  void reshapeOutMatSize();

  void reshapeBatchSize();

  // reshape input and output channel, height and width
  // layerSize == channel * height * width
  void reshapeImgSize();

  // FC do not change output layer size
  void setOutImgSize();

  void resetDnnBufferShapes();

  void resetDnnFwdPD(
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnFwdBuffers(const std::shared_ptr
    <mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnBotData(const std::shared_ptr
    <mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnTopData(const std::shared_ptr
    <mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetDnnWgtBiasData(const std::shared_ptr
    <mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void resetFwdPipeline(const std::shared_ptr
    <mkldnn::inner_product_forward::primitive_desc>& fwdPD);

  void forwardDnnVal();

  /*************************** for backward methods: **************************/
  void resetDnnBwdWgtPD(std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD);

  void resetDnnBwdDataPD(std::shared_ptr
    <mkldnn::inner_product_backward_data::primitive_desc>& bwdDataPD);

  void getBwdFwdPD(
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& bwdFwdPD);

  void resetDnnBwdBuffers(
    const std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD,
    const std::shared_ptr
    <mkldnn::inner_product_backward_data::primitive_desc>& bwdDataPD);

  void resetDnnTopDiffBwdData(const std::shared_ptr
    <mkldnn::inner_product_backward_data::primitive_desc>& bwdDataPD);

  void resetDnnTopDiffBwdWgt(const std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD);

  void resetDnnWgtBiasDiff(const std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD);

  void resetDnnBotDiff(const std::shared_ptr
    <mkldnn::inner_product_backward_data::primitive_desc>& bwdDataPD);

  void resetDnnBwdWgtPD(const std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD);

  void resetDnnBwdPipeline(
    const std::shared_ptr
    <mkldnn::inner_product_backward_weights::primitive_desc>& bwdWgtPD,
    const std::shared_ptr
    <mkldnn::inner_product_backward_data::primitive_desc>& bwdDataPD);

  void backwardDnnVal();

  void updateParameter(const UpdateCallback &callback);

  bool hasBotGrad();
};

}  // namespace paddle
