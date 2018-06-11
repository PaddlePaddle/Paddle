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

#include "ConvBaseOperator.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief ConvBaseOperator takes two inputs to perform the convolution.
 * The first input is the image, and the second input is the convolution kernel.
 * The height of data for two inputs are the same. Each data of the first input
 * is convolved with each data of the second input indepedently.
 *
 * The config file api is conv_operator.
 */

ConvBaseOperator::ConvBaseOperator(const OperatorConfig &config, bool useGpu)
    : Operator(config, useGpu) {
  CHECK(useGpu);
  CHECK_EQ(config_.input_indices_size(), 2L);

  caffeMode_ = true;
  getConvParams();
  computeConvSizes();

  // initialize all to default algorithms
  fwdAlgo_ = 0;
  bwdFilterAlgo_ = 0;
  bwdDataAlgo_ = 0;
  fwdLimitBytes_ = 0;
  bwdDataLimitBytes_ = 0;
  bwdFilterLimitBytes_ = 0;
  workSpaceInBytes_ = 0;
  workSpace_ = nullptr;

  isSelectAlgo_ = false;
}

void ConvBaseOperator::allocConvWorkSpace() {
  hl_conv_workspace(imageDesc_,
                    outputDesc_,
                    filterDesc_,
                    convDesc_,
                    &fwdAlgo_,
                    &fwdLimitBytes_,
                    &bwdDataAlgo_,
                    &bwdDataLimitBytes_,
                    &bwdFilterAlgo_,
                    &bwdFilterLimitBytes_,
                    /*useDilation*/ false);

  size_t maxWorkSpace = 0;
  maxWorkSpace = std::max(fwdLimitBytes_, bwdDataLimitBytes_);
  maxWorkSpace = std::max(maxWorkSpace, bwdFilterLimitBytes_);

  if (maxWorkSpace > workSpaceInBytes_) {
    if (workSpaceInBytes_ != 0) {
      hl_free_mem_device(workSpace_);
    }
    // total amount of storage needed
    workSpace_ = hl_malloc_device(maxWorkSpace);
    workSpaceInBytes_ = maxWorkSpace;
  }
}

void ConvBaseOperator::computeConvSizes() {
  hl_create_filter_descriptor(
      &filterDesc_, channels_, numFilters_, filterSizeY_, filterSize_);
  hl_create_tensor_descriptor(&imageDesc_);
  hl_create_tensor_descriptor(&outputDesc_);
  hl_create_convolution_descriptor(&convDesc_,
                                   imageDesc_,
                                   filterDesc_,
                                   paddingY_,
                                   padding_,
                                   strideY_,
                                   stride_);
}

void ConvBaseOperator::reshapeImageDescriptors() {
  hl_tensor_reshape(imageDesc_,
                    1,
                    channels_,
                    imageH_,
                    imageW_,
                    channels_ * imageH_ * imageW_,
                    imageH_ * imageW_,
                    imageW_,
                    1);
  hl_tensor_reshape(outputDesc_,
                    1,
                    numFilters_,
                    outputH_,
                    outputW_,
                    numFilters_ * outputH_ * outputW_,
                    outputH_ * outputW_,
                    outputW_,
                    1);
  hl_reset_convolution_descriptor(convDesc_,
                                  imageDesc_,
                                  filterDesc_,
                                  paddingY_,
                                  padding_,
                                  strideY_,
                                  stride_);
}

void ConvBaseOperator::getConvParams() {
  configNumFilters_ = config_.num_filters();
  const ConvConfig &conf = config_.conv_conf();
  padding_ = conf.padding();
  stride_ = conf.stride();
  filterSize_ = conf.filter_size();
  paddingY_ = conf.padding_y();
  strideY_ = conf.stride_y();
  filterSizeY_ = conf.filter_size_y();
  filterPixels_ = filterSize_ * filterSizeY_;
  configChannels_ = conf.channels();
  imgSize_ = conf.img_size();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  imgPixels_ = imgSize_ * imgSizeY_;
  CHECK_EQ(conf.groups(), 1U);
  filterChannels_ = conf.filter_channels();
  outputX_ = conf.output_x();
  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  outputs_ = outputX_ * outputX_;

  isDeconv_ = (config_.type() == "conv") ? false : true;
  if (isDeconv_) {
    channels_ = configNumFilters_;
    numFilters_ = configChannels_;
  } else {
    channels_ = configChannels_;
    numFilters_ = configNumFilters_;
  }
}

}  // namespace paddle
