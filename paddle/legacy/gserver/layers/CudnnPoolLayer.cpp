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

#include "CudnnPoolLayer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

bool CudnnPoolLayer::typeCheck(const std::string &poolType,
                               hl_pooling_mode_t *mode) {
  if (poolType == "cudnn-max-pool") {
    if (mode) {
      *mode = HL_POOLING_MAX;
    }
  } else if (poolType == "cudnn-avg-pool") {
    if (mode) {
      *mode = HL_POOLING_AVERAGE;
    }
  } else if (poolType == "cudnn-avg-incl-pad-pool") {
    if (mode) {
      *mode = HL_POOLING_AVERAGE_INCLUDE_PADDING;
    }
  } else {
    return false;
  }

  return true;
}

CudnnPoolLayer::CudnnPoolLayer(const LayerConfig &config) : PoolLayer(config) {
  const std::string &pool_type = config.inputs(0).pool_conf().pool_type();
  CHECK_EQ(CudnnPoolLayer::typeCheck(pool_type, &mode_), true);
}

bool CudnnPoolLayer::init(const LayerMap &layerMap,
                          const ParameterMap &parameterMap) {
  PoolLayer::init(layerMap, parameterMap);

  CHECK(useGpu_) << "CudnnPoolLayer only support gpu";

  hl_create_tensor_descriptor(&inputDesc_);
  hl_create_tensor_descriptor(&outputDesc_);

  windowHeight = sizeY_;
  windowWidth = sizeX_;
  heightPadding = confPaddingY_;
  widthPadding = confPadding_;
  strideHeight = strideY_;
  strideWidth = stride_;

  hl_create_pooling_descriptor(&poolingDesc_,
                               mode_,
                               windowHeight,
                               windowWidth,
                               heightPadding,
                               widthPadding,
                               strideHeight,
                               strideWidth);

  return true;
}

void CudnnPoolLayer::reshape(int batchSize) {
  imageH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imageW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imageH_ == 0) {
    imageH_ = imgSizeY_;
  }
  if (imageW_ == 0) {
    imageW_ = imgSize_;
  }
  CHECK_EQ(inputLayers_[0]->getOutput().value->getWidth(),
           channels_ * imageH_ * imageW_);
  outputH_ = outputSize(imageH_,
                        sizeY_,
                        confPaddingY_,
                        strideY_,
                        /* caffeMode */ false);
  outputW_ =
      outputSize(imageW_, sizeX_, confPadding_, stride_, /* caffeMode */ false);
  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);

  hl_tensor_reshape(inputDesc_, batchSize, channels_, imageH_, imageW_);
  hl_tensor_reshape(outputDesc_, batchSize, channels_, outputH_, outputW_);
}

void CudnnPoolLayer::forward(PassType passType) {
  Layer::forward(passType);

  CHECK(inputLayers_[0]->getOutputValue()->useGpu());
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  reshape(batchSize);
  resetOutput(batchSize, outputH_ * outputW_ * channels_);

  real *inputData = getInputValue(0)->getData();
  real *outData = getOutputValue()->getData();
  hl_pooling_forward(inputDesc_, inputData, outputDesc_, outData, poolingDesc_);
}

void CudnnPoolLayer::backward(const UpdateCallback &callback) {
  (void)callback;
  if (NULL == getInputGrad(0)) {
    return;
  }

  real *inputData = getInputValue(0)->getData();
  real *inputGrad = getInputGrad(0)->getData();
  real *outData = getOutputValue()->getData();
  real *outGrad = getOutputGrad()->getData();
  hl_pooling_backward(inputDesc_,
                      inputData,
                      inputGrad,
                      outputDesc_,
                      outData,
                      outGrad,
                      poolingDesc_);
}

CudnnPoolLayer::~CudnnPoolLayer() {
  hl_destroy_tensor_descriptor(inputDesc_);
  hl_destroy_tensor_descriptor(outputDesc_);
  hl_destroy_pooling_descriptor(poolingDesc_);
}

}  // namespace paddle
