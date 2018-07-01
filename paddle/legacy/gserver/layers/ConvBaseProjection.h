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

#include "Projection.h"
#include "paddle/math/MathUtils.h"

namespace paddle {

/**
 * @brief Base class for ConvProjection and ConvTransProjection.
 */
class ConvBaseProjection : public Projection {
 public:
  /**
   * Constructor.
   */
  ConvBaseProjection(const ProjectionConfig& config,
                     ParameterPtr parameter,
                     bool useGpu);

  ~ConvBaseProjection();

 protected:
  void getConvParams();
  void initCudnn();

  void reshapeTensorDesc(int batchSize);
  void reshape(int batchSize);

  virtual size_t calOutputSize() = 0;
  virtual size_t calInputSize() = 0;

  static void* getSpaceBytes(size_t size);

  /// True if it's deconv projection layer, false if it's ConvProjection layer
  bool isDeconv_;
  /// imageH_ and imageW_ / outputH_ and outputW_
  /// is calculated from the input layer.
  int imageH_, imageW_;
  int outputH_, outputW_;
  /// configImgH_ and configImgW_ / configOutH_ and configOutW_
  /// is obtained from config.
  int configImgH_, configImgW_;
  int configOutH_, configOutW_;
  /// channels_ and numFilters_ are defined in terms of convolution semantics
  int channels_, numFilters_;
  /// configChannels and configNumFilters_ are obtained from config
  /// For Conv they are the same as channels_ and numFilters
  /// For ConvTrans they are opposite to channels_ and numFilters
  int configChannels_, configNumFilters_;
  int paddingH_, paddingW_;
  int strideH_, strideW_;
  int dilationH_, dilationW_;
  int filterH_, filterW_;
  /// One group offset of input data.
  int inputOffset_;
  /// One group offset of output data.
  int outputOffset_;
  /// One group offset of weight.
  int weightOffset_;
  int groups_;

  /// Cudnn tensor descriptor for input.
  hl_tensor_descriptor imageDesc_;
  /// Cudnn tensor descriptor for output.
  hl_tensor_descriptor outputDesc_;
  /// Cudnn tensor descriptor for filter.
  hl_filter_descriptor filterDesc_;
  /// Cudnn tensor descriptor for a convolution operation.
  hl_convolution_descriptor convDesc_;

  /// Record the algorithm for forward convolution, which is obtained by cudnn
  /// api to search the best suited algorithm.
  int fwdAlgo_;
  /// Record the algorithm for computing convolution gradient with respect to
  /// filter coefficients.
  int bwdFilterAlgo_;
  /// Record the algorithm for computing convolution gradient with respect to
  /// the output.
  int bwdDataAlgo_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// forward convolution with the specified algo.
  size_t fwdLimitBytes_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// backwardFilter with the specified algo.
  size_t bwdDataLimitBytes_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// backwardData with the specified algo.
  size_t bwdFilterLimitBytes_;
  /// Size of total work space.
  size_t workSpaceInBytes_;
  bool bias_;

  std::unique_ptr<Weight> weight_;
  static ThreadLocalD<std::vector<MemoryHandlePtr>> convMem_;
};

}  // namespace paddle
