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

#include "ConvTransProjection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_PROJECTION(convt, ConvTransProjection);
size_t ConvTransProjection::calOutputSize() {
  outputH_ = in_->getFrameHeight();
  outputW_ = in_->getFrameWidth();
  if (outputH_ == 0) outputH_ = configOutH_;
  if (outputW_ == 0) outputW_ = configOutW_;
  imageH_ = imageSize(outputH_,
                      (filterH_ - 1) * dilationH_ + 1,
                      paddingH_,
                      strideH_,
                      /* caffeMode */ true);

  imageW_ = imageSize(outputW_,
                      (filterW_ - 1) * dilationW_ + 1,
                      paddingW_,
                      strideW_,
                      /* caffeMode */ true);

  const_cast<Argument *>(out_)->setFrameHeight(imageH_);
  const_cast<Argument *>(out_)->setFrameWidth(imageW_);

  inputOffset_ = (configChannels_ / groups_) * outputH_ * outputW_;
  outputOffset_ = (configNumFilters_ / groups_) * imageH_ * imageW_;
  return imageH_ * imageW_ * configNumFilters_;
}

size_t ConvTransProjection::calInputSize() {
  return static_cast<size_t>(configChannels_ * outputH_ * outputW_);
}

void ConvTransProjection::forward() {
  int batchSize = in_->value->getHeight();
  reshape(batchSize);

  void *workSpace = NULL;
  if (workSpaceInBytes_ > 0) {
    workSpace = getSpaceBytes(workSpaceInBytes_);
  }

  for (int g = 0; g < groups_; ++g) {
    REGISTER_TIMER_INFO("CudnnConvTransFwTimer", getName().c_str());

    real *inData = in_->value->getData() + g * inputOffset_;
    real *wgtData = weight_->getW()->getData() + g * weightOffset_;
    real *outData = out_->value->getData() + g * outputOffset_;
    hl_convolution_backward_data(imageDesc_,
                                 outData,
                                 outputDesc_,
                                 inData,
                                 filterDesc_,
                                 wgtData,
                                 convDesc_,
                                 workSpace,
                                 bwdDataLimitBytes_,
                                 bwdDataAlgo_);
  }
}

void ConvTransProjection::backward(const UpdateCallback &callback) {
  REGISTER_TIMER_INFO("CudnnConvTransBpTimer", getName().c_str());

  void *workSpace = NULL;
  if (workSpaceInBytes_ > 0) {
    workSpace = getSpaceBytes(workSpaceInBytes_);
  }

  for (int g = 0; g < groups_; ++g) {
    real *outGrad = out_->grad->getData() + g * outputOffset_;
    if (weight_->getWGrad()) {
      real *inData = in_->value->getData() + g * inputOffset_;
      real *weightGrad = weight_->getWGrad()->getData() + g * weightOffset_;
      hl_convolution_backward_filter(imageDesc_,
                                     outGrad,
                                     outputDesc_,
                                     inData,
                                     filterDesc_,
                                     weightGrad,
                                     convDesc_,
                                     workSpace,
                                     bwdFilterLimitBytes_,
                                     bwdFilterAlgo_);
    }

    MatrixPtr preGrad = in_->grad;
    if (NULL != preGrad) {
      real *inGrad = preGrad->getData() + g * inputOffset_;
      real *wgtData = weight_->getW()->getData() + g * weightOffset_;
      hl_convolution_forward(imageDesc_,
                             outGrad,
                             outputDesc_,
                             inGrad,
                             filterDesc_,
                             wgtData,
                             convDesc_,
                             workSpace,
                             fwdLimitBytes_,
                             fwdAlgo_);
    }
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
