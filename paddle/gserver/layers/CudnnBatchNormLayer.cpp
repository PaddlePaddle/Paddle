/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "CudnnBatchNormLayer.h"
#include "Layer.h"
#include "paddle/cuda/include/hl_batch_norm.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(cudnn_batch_norm, CudnnBatchNormLayer);

bool CudnnBatchNormLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!BatchNormBaseLayer::init(layerMap, parameterMap)) return false;
  CHECK(useGpu_) << "CudnnBatchNorm only support GPU";

  hl_create_tensor_descriptor(&ioDesc_);
  hl_create_tensor_descriptor(&bnParamDesc_);
  hl_tensor_reshape(bnParamDesc_, 1, channels_, 1, 1);

  return true;
}

void CudnnBatchNormLayer::reshape(int batchSize) {
  hl_tensor_reshape(ioDesc_, batchSize, channels_, imageH_ * imageD_, imageW_);
}

void CudnnBatchNormLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInputValue(0)->getHeight();
  calFeatureMapSize();
  reshape(batchSize);
  resetOutput(batchSize, getInputValue(0)->getWidth());

  // for testing in training peroid.
  useGlobalStats_ = (passType == PASS_TEST);
  if (passType == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }

  real* input = getInputValue(0)->getData();
  real* output = getOutputValue()->getData();
  real* gamma = weight_->getW()->getData();
  real* beta = biases_->getW()->getData();
  real* movingMean = movingMean_->getW()->getData();
  real* movingVar = movingVar_->getW()->getData();

  // cuDNN does not allow an epsilon value less than CUDNN_BN_MIN_EPSILON.
  eps_ = std::max(CUDNN_BN_MIN_EPSILON, static_cast<double>(epsilon_));

  if (!useGlobalStats_) {
    REGISTER_TIMER_INFO("CudnnBatchFwTimer", getName().c_str());
    real* savedMean = savedMean_->getData();
    real* savedInvVar = savedInvVar_->getData();
    hl_batch_norm_forward_training(ioDesc_,
                                   input,
                                   ioDesc_,
                                   output,
                                   bnParamDesc_,
                                   gamma,
                                   beta,
                                   1.0 - movingAvgFraction_,
                                   movingMean,
                                   movingVar,
                                   eps_,
                                   savedMean,
                                   savedInvVar);
  } else {
    // used movingMean and movingVar in testing
    if (batchSize <= 1024) {
      hl_batch_norm_forward_inference(ioDesc_,
                                      input,
                                      ioDesc_,
                                      output,
                                      bnParamDesc_,
                                      gamma,
                                      beta,
                                      movingMean,
                                      movingVar,
                                      eps_);
    } else {
      // There is a limitation in cudnn library.
      // When the batch size is larger than 1024 in cuDNN v5.1,
      // the cudnnBatchNormalizationForwardInference will fail.
      hl_batch_norm_cuda_inference(input,
                                   output,
                                   gamma,
                                   beta,
                                   movingMean,
                                   movingVar,
                                   eps_,
                                   batchSize,
                                   channels_,
                                   imageH_ * imageD_,
                                   imageW_);
    }
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void CudnnBatchNormLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  real* input = getInputValue(0)->getData();
  real* outGrad = getOutputGrad()->getData();
  real* inGrad = getInputGrad(0)->getData();
  real* gamma = weight_->getW()->getData();
  real* savedMean = savedMean_->getData();
  real* savedInvVar = savedInvVar_->getData();

  // cuDNN does not allow an epsilon value less than CUDNN_BN_MIN_EPSILON.
  eps_ = std::max(CUDNN_BN_MIN_EPSILON, static_cast<double>(epsilon_));

  auto create = [](MatrixPtr& m, size_t h, size_t w, real** p) {
    Matrix::resizeOrCreate(m, h, w, false, true);
    m->zeroMem();
    *p = m->getData();
  };

  real* gammaGrad = nullptr;
  real* betaGrad = nullptr;
  if (weight_->getWGrad()) {
    gammaGrad = weight_->getWGrad()->getData();
  } else {
    create(tmpWGrad_, 1, channels_, &gammaGrad);
  }
  if (biases_ && biases_->getWGrad()) {
    betaGrad = biases_->getWGrad()->getData();
  } else {
    create(tmpBiasGrad_, 1, channels_, &betaGrad);
  }

  hl_batch_norm_backward(ioDesc_,
                         input,
                         ioDesc_,
                         outGrad,
                         ioDesc_,
                         inGrad,
                         bnParamDesc_,
                         gamma,
                         gammaGrad,
                         betaGrad,
                         eps_,
                         savedMean,
                         savedInvVar);

  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    biases_->getParameterPtr()->incUpdate(callback);
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

CudnnBatchNormLayer::~CudnnBatchNormLayer() {
  hl_destroy_tensor_descriptor(ioDesc_);
  hl_destroy_tensor_descriptor(bnParamDesc_);
}

}  // namespace paddle
