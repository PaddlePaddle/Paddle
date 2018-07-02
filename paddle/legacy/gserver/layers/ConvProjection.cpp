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

#include "ConvProjection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_PROJECTION(conv, ConvProjection);

size_t ConvProjection::calOutputSize() {
  imageH_ = in_->getFrameHeight();
  imageW_ = in_->getFrameWidth();
  if (imageH_ == 0) imageH_ = configImgH_;
  if (imageW_ == 0) imageW_ = configImgW_;
  outputH_ = outputSize(imageH_,
                        (filterH_ - 1) * dilationH_ + 1,
                        paddingH_,
                        strideH_,
                        /* caffeMode */ true);
  outputW_ = outputSize(imageW_,
                        (filterW_ - 1) * dilationW_ + 1,
                        paddingW_,
                        strideW_,
                        /* caffeMode */ true);

  const_cast<Argument *>(out_)->setFrameHeight(outputH_);
  const_cast<Argument *>(out_)->setFrameWidth(outputW_);

  inputOffset_ = (configChannels_ / groups_) * imageH_ * imageW_;
  outputOffset_ = (configNumFilters_ / groups_) * outputH_ * outputW_;
  return outputH_ * outputW_ * configNumFilters_;
}

size_t ConvProjection::calInputSize() {
  return static_cast<size_t>(configChannels_ * imageH_ * imageW_);
}

void ConvProjection::forward() {
  int batchSize = in_->value->getHeight();
  reshape(batchSize);

  void *workSpace = NULL;
  if (workSpaceInBytes_ > 0) {
    workSpace = getSpaceBytes(workSpaceInBytes_);
  }

  for (int g = 0; g < groups_; ++g) {
    REGISTER_TIMER_INFO("CudnnConvFwTimer", getName().c_str());

    real *inputData = in_->value->getData() + g * inputOffset_;
    real *wgtData = weight_->getW()->getData() + g * weightOffset_;
    real *outData = out_->value->getData() + g * outputOffset_;
    hl_convolution_forward(imageDesc_,
                           inputData,
                           outputDesc_,
                           outData,
                           filterDesc_,
                           wgtData,
                           convDesc_,
                           workSpace,
                           fwdLimitBytes_,
                           fwdAlgo_);
  }
}

void ConvProjection::backward(const UpdateCallback &callback) {
  REGISTER_TIMER_INFO("CudnnConvBpTimer", getName().c_str());

  void *workSpace = NULL;
  if (workSpaceInBytes_ > 0) {
    workSpace = getSpaceBytes(workSpaceInBytes_);
  }

  for (int g = 0; g < groups_; ++g) {
    real *outGrad = out_->grad->getData() + g * outputOffset_;
    if (weight_->getWGrad()) {
      real *inputData = in_->value->getData() + g * inputOffset_;
      real *weightGrad = weight_->getWGrad()->getData() + g * weightOffset_;
      hl_convolution_backward_filter(imageDesc_,
                                     inputData,
                                     outputDesc_,
                                     outGrad,
                                     filterDesc_,
                                     weightGrad,
                                     convDesc_,
                                     workSpace,
                                     bwdFilterLimitBytes_,
                                     bwdFilterAlgo_);
    }

    MatrixPtr preGrad = in_->grad;
    if (NULL != preGrad) {
      real *inputGrad = preGrad->getData() + g * inputOffset_;
      real *wgtData = weight_->getW()->getData() + g * weightOffset_;
      hl_convolution_backward_data(imageDesc_,
                                   inputGrad,
                                   outputDesc_,
                                   outGrad,
                                   filterDesc_,
                                   wgtData,
                                   convDesc_,
                                   workSpace,
                                   bwdDataLimitBytes_,
                                   bwdDataAlgo_);
    }
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
