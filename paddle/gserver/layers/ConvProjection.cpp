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

#include "ConvProjection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_PROJECTION(conv, ConvProjection);

ThreadLocalD<std::vector<MemoryHandle *>> ConvProjection::convMem_;

ConvProjection::ConvProjection(const ProjectionConfig &config,
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

void ConvProjection::getConvParams() {
  const ConvConfig &conf = config_.conv_conf();
  paddingH_ = conf.padding_y();
  paddingW_ = conf.padding();

  strideH_ = conf.stride_y();
  strideW_ = conf.stride();

  filterH_ = conf.filter_size_y();
  filterW_ = conf.filter_size();

  configImgH_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  configImgW_ = conf.img_size();

  channels_ = conf.channels();
  numFilters_ = config_.num_filters();

  groups_ = conf.groups();
  CHECK_EQ(channels_ % groups_, 0);
  CHECK_EQ(numFilters_ % groups_, 0);
}

void ConvProjection::initCudnn() {
  hl_create_filter_descriptor(&filterDesc_,
                              channels_ / groups_,
                              numFilters_ / groups_,
                              filterH_,
                              filterW_);
  hl_create_tensor_descriptor(&inputDesc_);
  hl_create_tensor_descriptor(&outputDesc_);
  hl_create_convolution_descriptor(&convDesc_,
                                   inputDesc_,
                                   filterDesc_,
                                   paddingH_,
                                   paddingW_,
                                   strideH_,
                                   strideW_);

  // initialize all to default algorithms
  fwdAlgo_ = 0;
  bwdFilterAlgo_ = 0;
  bwdDataAlgo_ = 0;
  fwdLimitBytes_ = 0;
  bwdDataLimitBytes_ = 0;
  bwdFilterLimitBytes_ = 0;
  workSpaceInBytes_ = 0;

  batchNum_ = 0;
  isSelectAlgo_ = false;
}

void ConvProjection::reshapeTensorDesc(int batchSize) {
  hl_tensor_reshape(inputDesc_,
                    batchSize,
                    channels_ / groups_,
                    imageH_,
                    imageW_,
                    channels_ * imageH_ * imageW_,
                    imageH_ * imageW_,
                    imageW_,
                    1);
  hl_reset_convolution_descriptor(convDesc_,
                                  inputDesc_,
                                  filterDesc_,
                                  paddingH_,
                                  paddingW_,
                                  strideH_,
                                  strideW_);

  // The stride between two consecutive images in ConvProjection may not be 1,
  // for example, in the case of layer ConcatenateLayer2 with two
  // ConvProjection, the stride is the output_size of layer ConcatenateLayer2.
  // So the calculation of nStride is different from CudnnConvLayer.
  // In fact, only "nStride = out_->value->getStride()" is ok.
  size_t nStride = numFilters_ * outputH_ * outputW_;
  if (out_->value->isContiguous()) {
    CHECK_EQ(nStride, out_->value->getWidth());
  } else {
    nStride = out_->value->getStride();
  }

  hl_tensor_reshape(outputDesc_,
                    batchSize,
                    numFilters_ / groups_,
                    outputH_,
                    outputW_,
                    nStride,
                    outputH_ * outputW_,
                    outputW_,
                    1);
}

void ConvProjection::reshape(int batchSize) {
  size_t width = calOutputSize();
  CHECK_EQ(width, out_->value->getWidth());
  CHECK_EQ(static_cast<size_t>(channels_ * imageH_ * imageW_),
           in_->value->getWidth())
      << "Wrong input size for convolution"
      << " channels=" << channels_ << " imageH=" << imageH_
      << " imageW=" << imageW_ << " inputSize=" << in_->value->getWidth();

  isSelectAlgo_ = (batchSize == batchNum_);
  batchNum_ = batchSize;

  if (!isSelectAlgo_) {
    reshapeTensorDesc(batchSize);
    hl_conv_workspace(inputDesc_,
                      outputDesc_,
                      filterDesc_,
                      convDesc_,
                      &fwdAlgo_,
                      &fwdLimitBytes_,
                      &bwdDataAlgo_,
                      &bwdDataLimitBytes_,
                      &bwdFilterAlgo_,
                      &bwdFilterLimitBytes_);

    size_t maxWorkSpace = 0;
    maxWorkSpace = std::max(fwdLimitBytes_, bwdDataLimitBytes_);
    maxWorkSpace = std::max(maxWorkSpace, bwdFilterLimitBytes_);
    workSpaceInBytes_ = maxWorkSpace;

    VLOG(3) << getName() << " Fwd / BwdData / BwdFilter algo: " << fwdAlgo_
            << " / " << bwdDataAlgo_ << " / " << bwdFilterAlgo_;
  }

  isSelectAlgo_ = true;
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
    hl_convolution_forward(inputDesc_,
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
      hl_convolution_backward_filter(inputDesc_,
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
      hl_convolution_backward_data(inputDesc_,
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

void *ConvProjection::getSpaceBytes(size_t size) {
  std::vector<MemoryHandle *> &convMem = *convMem_;
  if (convMem.empty()) {
    int numDevices = hl_get_device_count();
    convMem.resize(numDevices);
  }

  int devId = hl_get_device();
  MemoryHandle **localMem = &(convMem[devId]);
  if (NULL == *localMem || size > (*localMem)->getAllocSize()) {
    *localMem = new GpuMemoryHandle(size);
  }
  return (*localMem)->getBuf();
}

ConvProjection::~ConvProjection() {
  hl_destroy_tensor_descriptor(inputDesc_);
  hl_destroy_tensor_descriptor(outputDesc_);
  hl_destroy_filter_descriptor(filterDesc_);
  hl_destroy_convolution_descriptor(convDesc_);
}

}  // namespace paddle
