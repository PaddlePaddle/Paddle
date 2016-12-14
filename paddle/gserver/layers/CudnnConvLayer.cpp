/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "CudnnConvLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(cudnn_conv, CudnnConvLayer);

bool CudnnConvLayer::init(const LayerMap &layerMap,
                          const ParameterMap &parameterMap) {
  if (!ConvBaseLayer::init(layerMap, parameterMap)) return false;
  CHECK(useGpu_) << "CudnnConvLayer only support gpu";

  CHECK_EQ(inputLayers_.size(), parameters_.size());
  projections_.reserve(inputLayers_.size());
  projConf_.reserve(inputLayers_.size());

  numFilters_ = config_.num_filters();
  CHECK(config_.shared_biases());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    ProjectionConfig *conf = new ProjectionConfig();
    conf->set_type("conv");
    conf->set_num_filters(numFilters_);
    ConvConfig *convConf = conf->mutable_conv_conf();
    *convConf = *(config_.mutable_inputs(i)->mutable_conv_conf());
    conf->set_input_size(getPrev(i)->getSize());
    conf->set_output_size(getSize());
    projConf_.emplace_back(conf);
    projections_.emplace_back(
        Projection::create(*projConf_[i], parameters_[i], useGpu_));
  }

  if (biases_.get() && sharedBiases_) {
    hl_create_tensor_descriptor(&biasDesc_);
    hl_create_tensor_descriptor(&outputDesc_);
    hl_tensor_reshape(biasDesc_, 1, numFilters_ / groups_[0], 1, 1);
    biasOffset_ = numFilters_ / groups_[0];
  }

  return true;
}

void CudnnConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  resetOutput(batchSize, calOutputSize());

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    projections_[i]->forward(&getInput(i), &getOutput(), passType);
  }

  if (biases_) {
    REGISTER_TIMER_INFO("CudnnConvBiasTimer", getName().c_str());
    int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
    hl_tensor_reshape(outputDesc_,
                      batchSize,
                      numFilters_ / groups_[0],
                      outputH_[0],
                      outputW_[0],
                      numFilters_ * outputH_[0] * outputW_[0],
                      outputH_[0] * outputW_[0],
                      outputW_[0],
                      1);
    outputOffset_ = getOutputValue()->getWidth() / groups_[0];
    for (int g = 0; g < groups_[0]; ++g) {
      real *biasData = biases_->getW()->getData() + biasOffset_ * g;
      real *outData = getOutputValue()->getData() + outputOffset_ * g;
      hl_convolution_forward_add_bias(
          biasDesc_, biasData, outputDesc_, outData);
    }
  }

  forwardActivation();
}

void CudnnConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("CudnnConvBpBiasTimer", getName().c_str());
    for (int g = 0; g < groups_[0]; ++g) {
      real *biasGrad = biases_->getWGrad()->getData() + biasOffset_ * g;
      real *outGrad = getOutputGrad()->getData() + outputOffset_ * g;
      hl_convolution_backward_bias(biasDesc_, biasGrad, outputDesc_, outGrad);
    }
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    projections_[i]->backward(callback);
  }
}

CudnnConvLayer::~CudnnConvLayer() {
  if (biases_) {
    hl_destroy_tensor_descriptor(biasDesc_);
    hl_destroy_tensor_descriptor(outputDesc_);
  }
}

}  // namespace paddle
