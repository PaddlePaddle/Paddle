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

#include "Weight.h"
#include "paddle/utils/Logging.h"

namespace paddle {

Weight::Weight(size_t height, size_t width, ParameterPtr param) {
  VectorPtr vPtr = param->getBuf(PARAMETER_VALUE);
  VectorPtr gPtr = param->getBuf(PARAMETER_GRADIENT);

  // create a new weight
  if (param->isSparse()) {
    CHECK_LE(param->getSize(), width * height);
  } else {
    CHECK_EQ(param->getSize(), width * height);
  }

  // weight_
  weight_ = param->getMat(PARAMETER_VALUE);
  if (!weight_ && vPtr) {
    weight_ = Matrix::create(vPtr->getMemoryHandle(), height, width);
  }
  if (weight_) {
    CHECK_EQ(height, weight_->getHeight());
    CHECK_EQ(width, weight_->getWidth());
  }

  // weightGrad
  weightGrad_ = param->getMat(PARAMETER_GRADIENT);
  if (!weightGrad_ && gPtr) {
    weightGrad_ = Matrix::create(gPtr->getMemoryHandle(), height, width);
  }
  if (weightGrad_) {
    CHECK_EQ(height, weightGrad_->getHeight());
    CHECK_EQ(width, weightGrad_->getWidth());
  }

  parameter_ = param;
}

Weight::Weight(size_t height, size_t width, ParameterPtr param, size_t offset) {
  VectorPtr vPtr = param->getBuf(PARAMETER_VALUE);
  VectorPtr gPtr = param->getBuf(PARAMETER_GRADIENT);

  // create a new weight
  CHECK_LE(offset + width * height, param->getSize());

  // weight_
  if (vPtr) {
    weight_ = Matrix::create(vPtr->getData() + offset,
                             height,
                             width,
                             /* trans */ false,
                             param->useGpu());
  }

  // weightGrad
  if (gPtr) {
    weightGrad_ = Matrix::create(gPtr->getData() + offset,
                                 height,
                                 width,
                                 /* trans */ false,
                                 param->useGpu());
  }

  parameter_ = param;
}

const ParameterPtr& Weight::getParameterPtr() { return parameter_; }
void Weight::setParameterPtr(ParameterPtr param) { parameter_ = param; }
}  // namespace paddle
