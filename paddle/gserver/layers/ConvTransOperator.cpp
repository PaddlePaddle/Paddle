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

#include "ConvTransOperator.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief ConvTransOperator takes two inputs to perform the convolution.
 * The first input is the image, and the second input is the convolution kernel.
 * The height of data for two inputs are the same. Each data of the first input
 * is convolved with each data of the second input indepedently.
 *
 * The config file api is conv_operator.
 */

REGISTER_OPERATOR(convt, ConvTransOperator);

void ConvTransOperator::forward() {
  size_t batchSize = ins_[0]->value->getHeight();
  reshape(batchSize);
  CHECK_EQ(ins_[1]->value->getHeight(), batchSize);
  checkFilterSize(ins_[1]->value);
  Matrix::resizeOrCreate(
      out_->value, batchSize, imageH_ * imageW_ * channels_, false, useGpu_);
  {
    AsyncGpuBlock block;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      real *inputData = ins_[0]->value->getData() + inputOffset_ * batchId;
      real *wgtData = ins_[1]->value->getData() + weightOffset_ * batchId;
      real *outData = out_->value->getData() + outputOffset_ * batchId;
      hl_convolution_backward_data(imageDesc_,
                                   outData,
                                   outputDesc_,
                                   inputData,
                                   filterDesc_,
                                   wgtData,
                                   convDesc_,
                                   workSpace_,
                                   workSpaceInBytes_,
                                   bwdDataAlgo_);
    }
  }
}

void ConvTransOperator::backward() {
  size_t batchSize = ins_[0]->value->getHeight();
  {
    AsyncGpuBlock block;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      real *outGrad = out_->grad->getData() + outputOffset_ * batchId;
      if (ins_[1]->grad) {
        real *inputData = ins_[0]->value->getData() + inputOffset_ * batchId;
        real *weightGrad = ins_[1]->grad->getData() + weightOffset_ * batchId;
        hl_convolution_backward_filter(imageDesc_,
                                       outGrad,
                                       outputDesc_,
                                       inputData,
                                       filterDesc_,
                                       weightGrad,
                                       convDesc_,
                                       workSpace_,
                                       workSpaceInBytes_,
                                       bwdFilterAlgo_);
      }

      MatrixPtr preGrad = ins_[0]->grad;
      if (NULL != preGrad) {
        real *inputGrad = preGrad->getData() + inputOffset_ * batchId;
        real *wgtData = ins_[1]->value->getData() + weightOffset_ * batchId;
        hl_convolution_forward(imageDesc_,
                               outGrad,
                               outputDesc_,
                               inputGrad,
                               filterDesc_,
                               wgtData,
                               convDesc_,
                               workSpace_,
                               workSpaceInBytes_,
                               fwdAlgo_);
      }
    }
  }
}

}  // namespace paddle
