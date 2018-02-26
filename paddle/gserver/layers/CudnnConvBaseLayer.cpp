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

#include "CudnnConvBaseLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {
REGISTER_LAYER(cudnn_conv, CudnnConvBaseLayer);
REGISTER_LAYER(cudnn_convt, CudnnConvBaseLayer);

bool CudnnConvBaseLayer::init(const LayerMap &layerMap,
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
    if (isDeconv_) {
      conf->set_type("convt");
    } else {
      conf->set_type("conv");
    }
    conf->set_num_filters(numFilters_);
    ConvConfig *convConf = conf->mutable_conv_conf();
    *convConf = *(config_.mutable_inputs(i)->mutable_conv_conf());
    conf->set_input_size(getPrev(i)->getSize());
    conf->set_output_size(getSize());
    projConf_.emplace_back(conf);
    projections_.emplace_back(
        Projection::create(*projConf_[i], parameters_[i], useGpu_));

    // create a new weight
    size_t height, width;
    height = filterPixels_[i] * filterChannels_[i];
    width = (!isDeconv_) ? numFilters_ : channels_[i];
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ =
          std::unique_ptr<Weight>(new Weight(numFilters_, 1, biasParameter_));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(getSize(), 1, biasParameter_));
    }
  }
  if (biases_.get() && sharedBiases_) {
    hl_create_tensor_descriptor(&biasDesc_);
    hl_create_tensor_descriptor(&outputDesc_);
    hl_tensor_reshape(biasDesc_, 1, numFilters_, 1, 1);
  }

  return true;
}

void CudnnConvBaseLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  resetOutput(batchSize, calOutputSize());

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    projections_[i]->forward(&getInput(i), &getOutput(), passType);
  }

  if (biases_) {
    REGISTER_TIMER_INFO("CudnnConvBiasTimer", getName().c_str());
    int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
    int outH = outputH_[0];
    int outW = outputW_[0];

    hl_tensor_reshape(outputDesc_,
                      batchSize,
                      numFilters_,
                      outH,
                      outW,
                      numFilters_ * outH * outW,
                      outH * outW,
                      outW,
                      1);
    real *outData = getOutputValue()->getData();
    real *biasData = biases_->getW()->getData();
    hl_convolution_forward_add_bias(biasDesc_, biasData, outputDesc_, outData);
  }

  forwardActivation();
}

void CudnnConvBaseLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("CudnnConvBpBiasTimer", getName().c_str());
    real *biasGrad = biases_->getWGrad()->getData();
    real *outGrad = getOutputGrad()->getData();
    hl_convolution_backward_bias(biasDesc_, biasGrad, outputDesc_, outGrad);

    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    projections_[i]->backward(callback);
  }
}

CudnnConvBaseLayer::~CudnnConvBaseLayer() {
  if (biases_) {
    hl_destroy_tensor_descriptor(biasDesc_);
    hl_destroy_tensor_descriptor(outputDesc_);
  }
}

}  // namespace paddle
