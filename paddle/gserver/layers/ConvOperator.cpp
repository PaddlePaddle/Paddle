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

class ConvOperator : public Operator {
public:
  ConvOperator(const OperatorConfig &config, bool useGpu);
  /**
   * Free workspace in device and destroy cudnn tensor descriptor.
   */
  virtual ~ConvOperator() {
    if (workSpaceInBytes_ != 0) {
      hl_free_mem_device(workSpace_);
      workSpaceInBytes_ = 0;
    }

    hl_destroy_tensor_descriptor(inputDesc_);
    hl_destroy_tensor_descriptor(outputDesc_);
    hl_destroy_filter_descriptor(filterDesc_);
    hl_destroy_convolution_descriptor(convDesc_);
  }
  virtual void forward();
  virtual void backward();

private:
  /**
   * Get convolution parameters from layer config and
   * initialize member variables.
   */
  void getConvParams();

  /**
   * Allocate Gpu Memory for cudnn convolution algorithms.
   */
  void allocConvWorkSpace(size_t maxWorkSpace);

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
  void reshape(int batchSize);

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
  int imageH_, imageW_, outputH_, outputW_;
  hl_tensor_descriptor inputDesc_;
  hl_tensor_descriptor outputDesc_;
  hl_filter_descriptor filterDesc_;
  hl_convolution_descriptor convDesc_;
  bool caffeMode_;
  int inputOffset_, outputOffset_, weightOffset_;
  int numFilters_;
  int padding_, stride_, filterSize_, channels_, imgSize_, imgSizeY_;
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

REGISTER_OPERATOR(conv, ConvOperator);

ConvOperator::ConvOperator(const OperatorConfig &config, bool useGpu)
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

void ConvOperator::allocConvWorkSpace(size_t maxWorkSpace) {
  if (maxWorkSpace > workSpaceInBytes_) {
    if (workSpaceInBytes_ != 0) {
      hl_free_mem_device(workSpace_);
    }
    // total amount of storage needed
    workSpace_ = hl_malloc_device(maxWorkSpace);
    workSpaceInBytes_ = maxWorkSpace;
  }
}

void ConvOperator::reshape(int batchSize) {
  imageH_ = ins_[0]->getFrameHeight();
  imageW_ = ins_[0]->getFrameWidth();
  if (imageH_ == 0) imageH_ = imgSizeY_;
  if (imageW_ == 0) imageW_ = imgSize_;
  outputH_ = outputSize(imageH_, filterSizeY_, paddingY_, strideY_, caffeMode_);
  outputW_ = outputSize(imageW_, filterSize_, padding_, stride_, caffeMode_);

  out_->setFrameHeight(outputH_);
  out_->setFrameWidth(outputW_);

  reshapeImageDescriptors();

  if (!isSelectAlgo_) {
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

    allocConvWorkSpace(maxWorkSpace);
  }

  isSelectAlgo_ = true;
}

void ConvOperator::computeConvSizes() {
  hl_create_filter_descriptor(
      &filterDesc_, channels_, numFilters_, filterSizeY_, filterSize_);
  hl_create_tensor_descriptor(&inputDesc_);
  int outputX =
      outputSize(imgSize_, filterSize_, padding_, stride_, caffeMode_);
  int outputY =
      outputSize(imgSizeY_, filterSizeY_, paddingY_, strideY_, caffeMode_);
  CHECK_EQ(outputX, outputX_);
  CHECK_EQ(outputY, outputY_);
  hl_create_tensor_descriptor(&outputDesc_);
  hl_create_convolution_descriptor(&convDesc_,
                                   inputDesc_,
                                   filterDesc_,
                                   paddingY_,
                                   padding_,
                                   strideY_,
                                   stride_);
}

void ConvOperator::reshapeImageDescriptors() {
  hl_tensor_reshape(inputDesc_,
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
                                  inputDesc_,
                                  filterDesc_,
                                  paddingY_,
                                  padding_,
                                  strideY_,
                                  stride_);
  inputOffset_ = channels_ * imageH_ * imageW_;
  outputOffset_ = numFilters_ * outputH_ * outputW_;
  weightOffset_ = numFilters_ * channels_ * filterSize_ * filterSize_;
}

void ConvOperator::getConvParams() {
  numFilters_ = config_.num_filters();
  const ConvConfig &conf = config_.conv_conf();
  padding_ = conf.padding();
  stride_ = conf.stride();
  filterSize_ = conf.filter_size();
  paddingY_ = conf.padding_y();
  strideY_ = conf.stride_y();
  filterSizeY_ = conf.filter_size_y();
  filterPixels_ = filterSize_ * filterSizeY_;
  channels_ = conf.channels();
  imgSize_ = conf.img_size();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  imgPixels_ = imgSize_ * imgSizeY_;
  CHECK_EQ(conf.groups(), 1U);
  filterChannels_ = conf.filter_channels();
  outputX_ = conf.output_x();
  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  outputs_ = outputX_ * outputX_;
}

void ConvOperator::forward() {
  size_t batchSize = ins_[0]->value->getHeight();
  reshape(batchSize);
  CHECK_EQ(ins_[1]->value->getHeight(), batchSize);
  checkFilterSize(ins_[1]->value);
  Matrix::resizeOrCreate(out_->value,
                         batchSize,
                         outputH_ * outputW_ * numFilters_,
                         false,
                         useGpu_);
  {
    AsyncGpuBlock block;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      real *inputData = ins_[0]->value->getData() + inputOffset_ * batchId;
      real *wgtData = ins_[1]->value->getData() + weightOffset_ * batchId;
      real *outData = out_->value->getData() + outputOffset_ * batchId;
      hl_convolution_forward(inputDesc_,
                             inputData,
                             outputDesc_,
                             outData,
                             filterDesc_,
                             wgtData,
                             convDesc_,
                             workSpace_,
                             workSpaceInBytes_,
                             fwdAlgo_);
    }
  }
}

void ConvOperator::backward() {
  size_t batchSize = ins_[0]->value->getHeight();
  {
    AsyncGpuBlock block;
    for (size_t batchId = 0; batchId < batchSize; ++batchId) {
      real *outGrad = out_->grad->getData() + outputOffset_ * batchId;
      if (ins_[1]->grad) {
        real *inputData = ins_[0]->value->getData() + inputOffset_ * batchId;
        real *weightGrad = ins_[1]->grad->getData() + weightOffset_ * batchId;
        hl_convolution_backward_filter(inputDesc_,
                                       inputData,
                                       outputDesc_,
                                       outGrad,
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
        hl_convolution_backward_data(inputDesc_,
                                     inputGrad,
                                     outputDesc_,
                                     outGrad,
                                     filterDesc_,
                                     wgtData,
                                     convDesc_,
                                     workSpace_,
                                     workSpaceInBytes_,
                                     bwdDataAlgo_);
      }
    }
  }
}

}  // namespace paddle
