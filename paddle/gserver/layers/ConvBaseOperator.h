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
#pragma once

#include "Operator.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief ConvOperator takes two inputs to perform the convolution.
 * The first input is the image, and the second input is the convolution kernel.
 * The height of data for two inputs are the same. Each data of the first input
 * is convolved with each data of the second input indepedently.
 *
 * The config file api is conv_operator.
 */

class ConvBaseOperator : public Operator {
 public:
  ConvBaseOperator(const OperatorConfig &config, bool useGpu);
  /**
   * Free workspace in device and destroy cudnn tensor descriptor.
   */
  virtual ~ConvBaseOperator() {
    if (workSpaceInBytes_ != 0) {
      hl_free_mem_device(workSpace_);
      workSpaceInBytes_ = 0;
    }

    hl_destroy_tensor_descriptor(imageDesc_);
    hl_destroy_tensor_descriptor(outputDesc_);
    hl_destroy_filter_descriptor(filterDesc_);
    hl_destroy_convolution_descriptor(convDesc_);
  }

 protected:
  /**
   * Get convolution parameters from layer config and
   * initialize member variables.
   */
  void getConvParams();

  /**
   * Allocate Gpu Memory for cudnn convolution algorithms.
   */
  void allocConvWorkSpace();

  /**
   * Create cudnn tensor descriptor for convolution operation.
   */
  void computeConvSizes();

  /**
   * Reshape cudnn tensor descriptor.
   */
  void reshapeImageDescriptors();

  /**
   * Reshape cudnn tensor descriptor.
   */
  virtual void reshape(int batchSize) = 0;

  /**
   * Check filter size is equal to the size calculated by parameters from
   * layer config.
   */
  void checkFilterSize(const MatrixPtr &filter) {
    CHECK_EQ(static_cast<int>(filter->getWidth()),
             filterSize_ * filterSizeY_ * channels_ * numFilters_);
  }

  /// Most of member variables are same with CudnnConvLayer.
  /// There is no explanation here.
  bool isDeconv_;
  int imageH_, imageW_, outputH_, outputW_;
  hl_tensor_descriptor imageDesc_;
  hl_tensor_descriptor outputDesc_;
  hl_filter_descriptor filterDesc_;
  hl_convolution_descriptor convDesc_;
  bool caffeMode_;
  int inputOffset_, outputOffset_, weightOffset_;
  int numFilters_, channels_;

  /// from parsing config
  int configNumFilters_, configChannels_;
  int padding_, stride_, filterSize_, imgSize_, imgSizeY_;
  int paddingY_, strideY_, filterSizeY_;
  int imgPixels_, filterPixels_, filterChannels_, outputX_, outputY_, outputs_;

  /// Following member variables are same with CudnnConvLayer.
  /// There is no explanation here.
  int fwdAlgo_, bwdFilterAlgo_, bwdDataAlgo_;
  size_t fwdLimitBytes_, bwdDataLimitBytes_, bwdFilterLimitBytes_;
  size_t workSpaceInBytes_;
  void *workSpace_;
  bool isSelectAlgo_;
};

}  // namespace paddle
