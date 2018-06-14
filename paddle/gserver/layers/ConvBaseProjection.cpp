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

#include "ConvBaseProjection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

ThreadLocalD<std::vector<MemoryHandlePtr>> ConvBaseProjection::convMem_;

ConvBaseProjection::ConvBaseProjection(const ProjectionConfig &config,
                                       ParameterPtr parameter,
                                       bool useGpu)
    : Projection(config, parameter, useGpu) {
  CHECK(useGpu);  // only support GPU
  getConvParams();
  initCudnn();

  size_t height = filterH_ * filterW_ * channels_ / groups_;
  size_t width = numFilters_;
  weight_.reset(new Weight(height, width, parameter));
  weightOffset_ = height * width / groups_;
}

void ConvBaseProjection::getConvParams() {
  const ConvConfig &conf = config_.conv_conf();
  paddingH_ = conf.padding_y();
  paddingW_ = conf.padding();

  strideH_ = conf.stride_y();
  strideW_ = conf.stride();

  dilationH_ = conf.dilation_y();
  dilationW_ = conf.dilation();
  CHECK_GT(dilationH_, 0);
  CHECK_GT(dilationW_, 0);

  filterH_ = conf.filter_size_y();
  filterW_ = conf.filter_size();

  configImgH_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  configImgW_ = conf.img_size();

  configOutH_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  configOutW_ = conf.output_x();

  configChannels_ = conf.channels();
  configNumFilters_ = config_.num_filters();

  isDeconv_ = (config_.type() == "conv") ? false : true;

  channels_ = (isDeconv_) ? configNumFilters_ : configChannels_;
  numFilters_ = (isDeconv_) ? configChannels_ : configNumFilters_;

  groups_ = conf.groups();
  CHECK_EQ(channels_ % groups_, 0);
  CHECK_EQ(numFilters_ % groups_, 0);
}

void ConvBaseProjection::initCudnn() {
  hl_create_filter_descriptor(&filterDesc_,
                              channels_ / groups_,
                              numFilters_ / groups_,
                              filterH_,
                              filterW_);
  hl_create_tensor_descriptor(&imageDesc_);
  hl_create_tensor_descriptor(&outputDesc_);
  hl_create_convolution_descriptor(&convDesc_,
                                   imageDesc_,
                                   filterDesc_,
                                   paddingH_,
                                   paddingW_,
                                   strideH_,
                                   strideW_,
                                   dilationH_,
                                   dilationW_);

  // initialize all to default algorithms
  fwdAlgo_ = 0;
  bwdFilterAlgo_ = 0;
  bwdDataAlgo_ = 0;
  fwdLimitBytes_ = 0;
  bwdDataLimitBytes_ = 0;
  bwdFilterLimitBytes_ = 0;
  workSpaceInBytes_ = 0;
}

void ConvBaseProjection::reshapeTensorDesc(int batchSize) {
  // The stride between two consecutive samples in the output of ConvProjection
  // may not be numFilters_ * outputH_ * outputW_ (conv) or
  // channels_ * imageH_ * imageW_ (deconv)
  // for example, in the case of layer ConcatenateLayer2 with two
  // ConvProjection, the stride is the output_size of layer ConcatenateLayer2.
  // So the calculation of nStride is different from CudnnConvLayer.
  size_t nStrideImage, nStrideOutput;
  if (isDeconv_) {
    nStrideImage = out_->value->getStride();
    nStrideOutput = numFilters_ * outputH_ * outputW_;
  } else {
    nStrideImage = channels_ * imageH_ * imageW_;
    nStrideOutput = out_->value->getStride();
  }

  hl_tensor_reshape(imageDesc_,
                    batchSize,
                    channels_ / groups_,
                    imageH_,
                    imageW_,
                    nStrideImage,
                    imageH_ * imageW_,
                    imageW_,
                    1);

  hl_tensor_reshape(outputDesc_,
                    batchSize,
                    numFilters_ / groups_,
                    outputH_,
                    outputW_,
                    nStrideOutput,
                    outputH_ * outputW_,
                    outputW_,
                    1);

  hl_reset_convolution_descriptor(convDesc_,
                                  imageDesc_,
                                  filterDesc_,
                                  paddingH_,
                                  paddingW_,
                                  strideH_,
                                  strideW_,
                                  dilationH_,
                                  dilationW_);
}

void ConvBaseProjection::reshape(int batchSize) {
  size_t width = calOutputSize();
  CHECK_EQ(width, out_->value->getWidth());
  CHECK_EQ(calInputSize(), in_->value->getWidth());

  reshapeTensorDesc(batchSize);
  bool useDilation = false;
  if (dilationH_ > 1 || dilationW_ > 1) {
    useDilation = true;
  }
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
                    useDilation);

  size_t maxWorkSpace = 0;
  maxWorkSpace = std::max(fwdLimitBytes_, bwdDataLimitBytes_);
  maxWorkSpace = std::max(maxWorkSpace, bwdFilterLimitBytes_);
  workSpaceInBytes_ = maxWorkSpace;

  VLOG(3) << getName() << " Fwd / BwdData / BwdFilter algo: " << fwdAlgo_
          << " / " << bwdDataAlgo_ << " / " << bwdFilterAlgo_;
}

void *ConvBaseProjection::getSpaceBytes(size_t size) {
  std::vector<MemoryHandlePtr> &convMem = *convMem_;
  if (convMem.empty()) {
    int numDevices = hl_get_device_count();
    convMem.resize(numDevices);
  }

  int devId = hl_get_device();
  MemoryHandlePtr localMem = convMem[devId];
  if (NULL == localMem || size > localMem->getAllocSize()) {
    localMem = std::make_shared<GpuMemoryHandle>(size);
  }
  return localMem->getBuf();
}

ConvBaseProjection::~ConvBaseProjection() {
  hl_destroy_tensor_descriptor(imageDesc_);
  hl_destroy_tensor_descriptor(outputDesc_);
  hl_destroy_filter_descriptor(filterDesc_);
  hl_destroy_convolution_descriptor(convDesc_);
}

}  // namespace paddle
