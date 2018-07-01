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

#include "Pool3DLayer.h"
#include "PoolProjectionLayer.h"
#include "paddle/utils/Logging.h"

namespace paddle {

REGISTER_LAYER(pool3d, Pool3DLayer);

bool Pool3DLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
  poolType_ = conf.pool_type();
  channels_ = conf.channels();

  sizeX_ = conf.size_x();
  sizeY_ = conf.size_y();
  sizeZ_ = conf.size_z();

  strideW_ = conf.stride();
  strideH_ = conf.stride_y();
  strideD_ = conf.stride_z();

  imgSizeW_ = conf.img_size();
  imgSizeH_ = conf.img_size_y();
  imgSizeD_ = conf.img_size_z();

  paddingW_ = conf.padding();
  paddingH_ = conf.padding_y();
  paddingD_ = conf.padding_z();

  outputW_ = conf.output_x();
  outputH_ = conf.output_y();
  outputD_ = conf.output_z();

  return true;
}

size_t Pool3DLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);

  size_t layerSize = 0;
  outputD_ = outputSize(imgSizeD_, sizeZ_, paddingD_, strideD_, false);
  outputH_ = outputSize(imgSizeH_, sizeY_, paddingH_, strideH_, false);
  outputW_ = outputSize(imgSizeW_, sizeX_, paddingW_, strideW_, false);

  layerSize = outputD_ * outputH_ * outputW_ * channels_;
  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);
  getOutput().setFrameDepth(outputD_);
  return layerSize;
}

void Pool3DLayer::forward(PassType passType) {
  Layer::forward(passType);
  const MatrixPtr& inMat = inputLayers_[0]->getOutputValue();
  size_t batchSize = inMat->getHeight();
  size_t outWidth = getSize();
  resetOutput(batchSize, outWidth);
  Matrix::resizeOrCreate(maxPoolIdx_, batchSize, outWidth, false, useGpu_);
  const MatrixPtr outMat = getOutputValue();

  if (poolType_ == "avg") {
    outMat->avgPool3DForward(*inMat,
                             channels_,
                             imgSizeD_,
                             imgSizeH_,
                             imgSizeW_,
                             outputD_,
                             outputH_,
                             outputW_,
                             sizeZ_,
                             sizeY_,
                             sizeX_,
                             strideD_,
                             strideH_,
                             strideW_,
                             paddingD_,
                             paddingH_,
                             paddingW_);
  } else if (poolType_ == "max") {
    outMat->maxPool3DForward(*inMat,
                             *maxPoolIdx_,
                             channels_,
                             imgSizeD_,
                             imgSizeH_,
                             imgSizeW_,
                             outputD_,
                             outputH_,
                             outputW_,
                             sizeZ_,
                             sizeY_,
                             sizeX_,
                             strideD_,
                             strideH_,
                             strideW_,
                             paddingD_,
                             paddingH_,
                             paddingW_);
  } else {
    LOG(FATAL) << "Unknown pool type: " << poolType_;
  }
  forwardActivation();
}

void Pool3DLayer::backward(const UpdateCallback& callback) {
  backwardActivation();

  (void)callback;
  if (NULL == getInputGrad(0)) return;
  MatrixPtr inMat = inputLayers_[0]->getOutputValue();
  MatrixPtr inGradMat = inputLayers_[0]->getOutputGrad();
  MatrixPtr outMat = getOutputValue();
  MatrixPtr outGradMat = getOutputGrad();

  if (poolType_ == "avg") {
    inGradMat->avgPool3DBackward(*outGradMat,
                                 imgSizeD_,
                                 imgSizeH_,
                                 imgSizeW_,
                                 outputD_,
                                 outputH_,
                                 outputW_,
                                 sizeZ_,
                                 sizeY_,
                                 sizeZ_,
                                 strideD_,
                                 strideH_,
                                 strideW_,
                                 paddingD_,
                                 paddingH_,
                                 paddingW_,
                                 1.0,
                                 1.0);
  } else if (poolType_ == "max") {
    inGradMat->maxPool3DBackward(*outGradMat,
                                 *maxPoolIdx_,
                                 imgSizeD_,
                                 imgSizeH_,
                                 imgSizeW_,
                                 outputD_,
                                 outputH_,
                                 outputW_,
                                 sizeZ_,
                                 sizeY_,
                                 sizeZ_,
                                 strideD_,
                                 strideH_,
                                 strideW_,
                                 paddingD_,
                                 paddingH_,
                                 paddingW_,
                                 1.0,
                                 1.0);
  } else {
    LOG(FATAL) << "Unknown pool type: " << poolType_;
  }
}

}  // namespace paddle
