/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

REGISTER_PROJECTION_CREATE_FUNC(pool2, &PoolProjection::create);

PoolProjection* PoolProjection::create(const ProjectionConfig& config,
                                       ParameterPtr parameter, bool useGpu) {
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
  MatrixPtr inputV = in_->value;
  MatrixPtr outV = out_->value;
  outV->maxPoolForward(*inputV, imgSizeY_, imgSize_, channels_,
                       sizeX_, sizeY_, strideY_, stride_,
                       outputY_, outputX_, confPaddingY_, confPadding_);
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
  inputGrad->maxPoolBackward(*inputV, imgSizeY_, imgSize_, *outGrad, *outV,
                             sizeX_, sizeY_,
                             strideY_, stride_, outputY_, outputX_, 1, 1,
                             confPaddingY_, confPadding_);
}

void AvgPoolProjection::forward() {
  MatrixPtr inputV = in_->value;
  MatrixPtr outV = out_->value;
  outV->avgPoolForward(*inputV, imgSizeY_, imgSize_, channels_,
                       sizeX_, sizeY_, strideY_, stride_,
                       outputY_, outputX_, confPaddingY_, confPadding_);
}

void AvgPoolProjection::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = out_->grad;
  MatrixPtr inputGrad = in_->grad;

  if (NULL == inputGrad) {
    return;
  }

  inputGrad->avgPoolBackward(*outputGrad, imgSizeY_, imgSize_,
                             sizeX_, sizeY_, strideY_, stride_,
                             outputY_, outputX_, 1, 1,
                             confPaddingY_, confPadding_);
}
}  // namespace paddle
