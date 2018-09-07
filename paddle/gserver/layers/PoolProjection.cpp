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

#include "PoolProjection.h"

namespace paddle {

REGISTER_PROJECTION_CREATE_FUNC(pool, &PoolProjection::create);

PoolProjection::PoolProjection(const ProjectionConfig& config,
                               ParameterPtr parameter,
                               bool useGpu)
    : Projection(config, parameter, useGpu) {
  const PoolConfig& conf = config_.pool_conf();
  poolType_ = conf.pool_type();
  channels_ = conf.channels();
  sizeX_ = conf.size_x();
  stride_ = conf.stride();
  outputX_ = conf.output_x();
  imgSize_ = conf.img_size();
  confPadding_ = conf.padding();

  sizeY_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  strideY_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  confPaddingY_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();

  excludeMode_ = conf.has_exclude_mode() ? conf.exclude_mode() : true;
}

size_t PoolProjection::getSize() {
  imgSizeY_ = in_->getFrameHeight();
  imgSize_ = in_->getFrameWidth();
  const PoolConfig& conf = config_.pool_conf();
  if (imgSizeY_ == 0) {
    imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  }
  if (imgSize_ == 0) {
    imgSize_ = conf.img_size();
  }
  outputY_ = outputSize(imgSizeY_,
                        sizeY_,
                        confPaddingY_,
                        strideY_,
                        /* caffeMode */ false);
  outputX_ = outputSize(imgSize_,
                        sizeX_,
                        confPadding_,
                        stride_,
                        /* caffeMode */ false);

  const_cast<Argument*>(out_)->setFrameHeight(outputY_);
  const_cast<Argument*>(out_)->setFrameWidth(outputX_);

  return outputY_ * outputX_ * channels_;
}

PoolProjection* PoolProjection::create(const ProjectionConfig& config,
                                       ParameterPtr parameter,
                                       bool useGpu) {
  const std::string& pool = config.pool_conf().pool_type();
  if (pool == "max-projection") {
    return new MaxPoolProjection(config, parameter, useGpu);
  } else if (pool == "avg-projection") {
    return new AvgPoolProjection(config, parameter, useGpu);
  } else {
    LOG(FATAL) << "Unknown pool type: " << pool;
    return nullptr;
  }
}

void MaxPoolProjection::forward() {
  size_t width = getSize();
  CHECK_EQ(width, out_->value->getWidth());
  MatrixPtr inputV = in_->value;
  MatrixPtr outV = out_->value;
  outV->maxPoolForward(*inputV,
                       imgSizeY_,
                       imgSize_,
                       channels_,
                       sizeX_,
                       sizeY_,
                       strideY_,
                       stride_,
                       outputY_,
                       outputX_,
                       confPaddingY_,
                       confPadding_);
}

void MaxPoolProjection::backward(const UpdateCallback& callback) {
  (void)callback;
  MatrixPtr outGrad = out_->grad;
  MatrixPtr inputV = in_->value;
  MatrixPtr outV = out_->value;
  MatrixPtr inputGrad = in_->grad;

  if (NULL == inputGrad) {
    return;
  }
  inputGrad->maxPoolBackward(*inputV,
                             imgSizeY_,
                             imgSize_,
                             *outGrad,
                             *outV,
                             sizeX_,
                             sizeY_,
                             strideY_,
                             stride_,
                             outputY_,
                             outputX_,
                             1,
                             1,
                             confPaddingY_,
                             confPadding_);
}

void AvgPoolProjection::forward() {
  size_t width = getSize();
  CHECK_EQ(width, out_->value->getWidth());
  MatrixPtr inputV = in_->value;
  MatrixPtr outV = out_->value;
  outV->avgPoolForward(*inputV,
                       imgSizeY_,
                       imgSize_,
                       channels_,
                       sizeX_,
                       sizeY_,
                       strideY_,
                       stride_,
                       outputY_,
                       outputX_,
                       confPaddingY_,
                       confPadding_,
                       excludeMode_);
}

void AvgPoolProjection::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = out_->grad;
  MatrixPtr inputGrad = in_->grad;

  if (NULL == inputGrad) {
    return;
  }

  inputGrad->avgPoolBackward(*outputGrad,
                             imgSizeY_,
                             imgSize_,
                             sizeX_,
                             sizeY_,
                             strideY_,
                             stride_,
                             outputY_,
                             outputX_,
                             1,
                             1,
                             confPaddingY_,
                             confPadding_,
                             excludeMode_);
}
}  // namespace paddle
