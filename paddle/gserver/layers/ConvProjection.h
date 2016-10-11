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


#pragma once

#include "Projection.h"

namespace paddle {

/**
 * @brief Convolution projection do the same calculation with CudnnConvLayer.
 */
class ConvProjection : public Projection {
public:
  /**
   * Constructor.
   */
  ConvProjection(const ProjectionConfig& config, ParameterPtr parameter,
                 bool useGpu);
  virtual void forward();
  virtual void backward(const UpdateCallback& callback);

  int getChannels() {
    return channels_;
  }

  void createBias(int elmCnt) {
    hl_create_tensor_descriptor(&allOutputDesc_);
    hl_create_tensor_descriptor(&biasDesc_);
    hl_tensor_reshape(biasDesc_, 1, elmCnt, 1, 1);
  }

  void addBias(int batchSize, int numFilters, real* outData, real* biasData) {
    CHECK(allOutputDesc_ && biasDesc_);
    // numFilters != numFilters_ in case of concat2(input=[convProj, convProj])
    hl_tensor_reshape(allOutputDesc_, batchSize, numFilters,
                      outputH_, outputW_, numFilters * outputH_ * outputW_,
                      outputH_ * outputW_, outputW_, 1);

    hl_convolution_forward_add_bias(biasDesc_, biasData,
                                    allOutputDesc_, outData);
  }

  void bpropBias(real* outGrad, real* biasGrad) {
    CHECK(allOutputDesc_ && biasDesc_);
    hl_convolution_backward_bias(biasDesc_, biasGrad, outputDesc_, outGrad);
  }

protected:
  void getConvParams();
  void initCudnn();

  void allocConvWorkSpace(size_t maxWorkSpace);

  void reshapeTensorDesc(int batchSize);

  void reshape(int batchSize);

  int outputSize(int imageSize, int filterSize, int padding, int stride) {
    // caffe mode
    int outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
    return outputSize;
  }

  /// imageH_ and imageW_ is calculated from the input layer.
  int imageH_, imageW_;
  /// configImgH_ and configImgW_ is obtained from config.
  int configImgH_, configImgW_;
  int outputH_, outputW_;
  int channels_, numFilters_;
  int paddingH_, paddingW_;
  int strideH_, strideW_;
  int filterH_, filterW_;
  int inputOffset_, outputOffset_, weightOffset_;

  hl_tensor_descriptor inputDesc_;
  hl_tensor_descriptor outputDesc_;
  hl_filter_descriptor filterDesc_;
  hl_convolution_descriptor convDesc_;

  /// if ConvProjection is the input of ConcatenateLayer2,
  /// this projection's output is only sub columns of
  /// ConcatenateLayer2's output matrix.
  /// allOutputDesc_ is used to describe the ConcatenateLayer2's output.
  /// It is used when the ConcatenateLayer2 has bias.
  hl_tensor_descriptor allOutputDesc_;
  hl_tensor_descriptor biasDesc_;

  /// Following member variables are same with CudnnConvLayer.
  /// There is no explanation here.
  int fwdAlgo_, bwdFilterAlgo_, bwdDataAlgo_;
  size_t fwdLimitBytes_, bwdDataLimitBytes_, bwdFilterLimitBytes_;
  size_t workSpaceInBytes_;
  void* workSpace_;

  bool isSelectAlgo_;
  /// batchNum is used to record batch size. If the batch size is changed,
  /// the selection algorithm will be called.
  int batchNum_;

  std::unique_ptr<Weight> weight_;
};

}  // namespace paddle
