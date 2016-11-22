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


#include "paddle/utils/Stat.h"
#include "ConvProjection.h"

namespace paddle {

REGISTER_PROJECTION(conv, ConvProjection);

ConvProjection::ConvProjection(const ProjectionConfig& config,
                               ParameterPtr parameter, bool useGpu)
    : Projection(config, parameter, useGpu), bias_(false) {

  CHECK(useGpu);  // only support GPU
  getConvParams();
  initCudnn();

  size_t height = filterH_ * filterW_ * channels_;
  size_t width = numFilters_;
  weight_.reset(new Weight(height, width, parameter));
}

void ConvProjection::getConvParams() {
  const ConvConfig &conf = config_.conv_conf();

  paddingH_ = conf.padding_y();
  paddingW_ = conf.padding();

  strideH_ = conf.stride_y();
  strideW_ = conf.stride();

  filterH_ = conf.filter_size_y();
  filterW_ = conf.filter_size();

  configImgH_ = conf.img_size();
  configImgW_ = conf.img_size();

  channels_ = conf.channels();
  numFilters_ = config_.num_filters();

  CHECK_EQ(conf.groups(), 1U);
}

void ConvProjection::initCudnn() {
  hl_create_filter_descriptor(&filterDesc_, channels_, numFilters_,
                              filterH_, filterW_);
  hl_create_tensor_descriptor(&inputDesc_);
  hl_create_tensor_descriptor(&outputDesc_);
  hl_create_convolution_descriptor(&convDesc_, inputDesc_, filterDesc_,
                                   paddingH_, paddingW_, strideH_, strideW_);

  // initialize all to default algorithms
  fwdAlgo_ = 0;
  bwdFilterAlgo_ = 0;
  bwdDataAlgo_ = 0;
  fwdLimitBytes_ = 0;
  bwdDataLimitBytes_ = 0;
  bwdFilterLimitBytes_ = 0;
  workSpaceInReal_ = 0;

  batchNum_ = 0;
  isSelectAlgo_ = false;
}

void ConvProjection::reshapeTensorDesc(int batchSize) {
  hl_tensor_reshape(inputDesc_, batchSize, channels_, imageH_, imageW_,
                    channels_ * imageH_ * imageW_, imageH_ * imageW_,
                    imageW_, 1);
  hl_reset_convolution_descriptor(convDesc_, inputDesc_, filterDesc_,
                                  paddingH_, paddingW_, strideH_, strideW_);

  // The stride between two consecutive images in ConvProjection may not be 1,
  // for example, in the case of layer ConcatenateLayer2 with two
  // ConvProjection, the stride is the output_size of layer ConcatenateLayer2.
  // So the calculation of nStride is different from CudnnConvLayer.
  size_t nStride = numFilters_ * outputH_ * outputW_;
  if (out_->value->isContiguous()) {
    CHECK_EQ(nStride, out_->value->getWidth());
  } else {
    nStride = out_->value->getStride();
  }

  hl_tensor_reshape(outputDesc_, batchSize, numFilters_, outputH_, outputW_,
                    nStride, outputH_ * outputW_, outputW_, 1);
}

void ConvProjection::reshape(int batchSize) {
  calOutputSize();
  isSelectAlgo_ = (batchSize == batchNum_);
  batchNum_ = batchSize;

  if (!isSelectAlgo_) {
    reshapeTensorDesc(batchSize);
    hl_conv_workspace(inputDesc_, outputDesc_, filterDesc_,
                      convDesc_, &fwdAlgo_, &fwdLimitBytes_,
                      &bwdDataAlgo_, &bwdDataLimitBytes_,
                      &bwdFilterAlgo_, &bwdFilterLimitBytes_);

    size_t maxWorkSpace = 0;
    maxWorkSpace = std::max(fwdLimitBytes_, bwdDataLimitBytes_);
    maxWorkSpace = std::max(maxWorkSpace, bwdFilterLimitBytes_);
    workSpaceInReal_ = (maxWorkSpace + sizeof(real) - 1) / sizeof(real);


    VLOG(3) << getName() << " Fwd / BwdData / BwdFilter algo: " << fwdAlgo_
                         << " / " << bwdDataAlgo_
                         << " / " << bwdFilterAlgo_;
  }

  isSelectAlgo_ = true;
}

void ConvProjection::forward() {
  int batchSize = in_->value->getHeight();
  reshape(batchSize);

  real *inputData = in_->value->getData();
  real *wgtData = weight_->getW()->getData();
  real *outData = out_->value->getData();

  void* workSpace = NULL;
  if (workSpaceInReal_ > 0) {
    MatrixPtr tmpMat = Matrix::getTmpMatrix(1, workSpaceInReal_, true);
    workSpace = (void*)tmpMat->getData();
  }

  REGISTER_TIMER_INFO("ConvProjectionFwTimer", getName().c_str());
  hl_convolution_forward(inputDesc_, inputData, outputDesc_,
                         outData, filterDesc_, wgtData,
                         convDesc_, workSpace,
                         fwdLimitBytes_, fwdAlgo_);
}

void ConvProjection::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("ConvProjectionBpTimer", getName().c_str());

  void* workSpace = NULL;
  if (workSpaceInReal_ > 0) {
    MatrixPtr tmpMat = Matrix::getTmpMatrix(1, workSpaceInReal_, true);
    workSpace = (void*)tmpMat->getData();
  }

  real *outGrad = out_->grad->getData();
  if (weight_->getWGrad()) {
    real *inputData = in_->value->getData();
    real *weightGrad = weight_->getWGrad()->getData();
    hl_convolution_backward_filter(
        inputDesc_, inputData, outputDesc_, outGrad, filterDesc_,
        weightGrad, convDesc_, workSpace, bwdFilterLimitBytes_,
        bwdFilterAlgo_);
  }

  MatrixPtr preGrad = in_->grad;
  if (NULL != preGrad) {
    real *inputGrad = preGrad->getData();
    real *wgtData = weight_->getW()->getData();
    hl_convolution_backward_data(
        inputDesc_, inputGrad, outputDesc_, outGrad, filterDesc_,
        wgtData, convDesc_, workSpace, bwdDataLimitBytes_,
        bwdDataAlgo_);
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

ConvProjection::~ConvProjection() {
  hl_destroy_tensor_descriptor(inputDesc_);
  hl_destroy_tensor_descriptor(outputDesc_);
  hl_destroy_filter_descriptor(filterDesc_);
  hl_destroy_convolution_descriptor(convDesc_);

  if (bias_) {
    hl_destroy_tensor_descriptor(allOutputDesc_);
    hl_destroy_tensor_descriptor(biasDesc_);
  }
}

}  // namespace paddle
