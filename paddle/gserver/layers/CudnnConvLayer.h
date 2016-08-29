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

#include "ConvBaseLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>

namespace paddle {

/**
 * @brief A subclass of ConvBaseLayer by cuDNN implementation. It only
 *        supports GPU mode. We automatic select CudnnConvLayer for GPU
 *        mode and ExpandConvLayer for CPU mode if you set type of "conv".
 *        User also can specfiy type of "exconv" or "cudnn_conv" for
 *        particular type.
 *
 * The config file api is img_conv_layer.
 */
class CudnnConvLayer : public ConvBaseLayer {
private:
  /// resize Cudnn workspace size
  void allocConvWorkSpace(size_t maxWorkSpace);

protected:
  int imageH_, imageW_, outputH_, outputW_;
  /// Cudnn tensor descriptor for bias.
  hl_tensor_descriptor biasDesc_;
  /// Cudnn tensor descriptor for input.
  std::vector<hl_tensor_descriptor> inputDesc_;
  /// Cudnn tensor descriptor for output.
  std::vector<hl_tensor_descriptor> outputDesc_;
  /// Cudnn tensor descriptor for filter.
  std::vector<hl_filter_descriptor> filterDesc_;
  /// Cudnn tensor descriptor for a convolution operation.
  std::vector<hl_convolution_descriptor> convDesc_;
  /// One sample offset of input data.
  IntV inputOffset_;
  /// One sample offset of output data.
  IntV outputOffset_;
  /// One group offset of weight.
  IntV weightOffset_;
  /// One group offset of bias.
  int biasOffset_;

  /// Save the algorithm for forward convolution, which is obtained by cudnn
  /// api to search the best suited algorithm.
  std::vector<int> fwdAlgo_;
  /// Save the algorithm for computing convolution gradient with respect to
  /// filter coefficients.
  std::vector<int> bwdFilterAlgo_;
  /// Save the algorithm for computing convolution gradient with respect to
  /// the output.
  std::vector<int> bwdDataAlgo_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// forward convolution with the specified algo.
  std::vector<size_t> fwdLimitBytes_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// backwardFilter with the specified algo.
  std::vector<size_t> bwdFilterLimitBytes_;
  /// Amount of GPU memory needed as workspace to be able to execute a
  /// backwardData with the specified algo.
  std::vector<size_t> bwdDataLimitBytes_;

  /// Device work space address for each group.
  std::vector<void*> workSpace_;
  /// Max number of groups.
  int maxGroups_;
  /// Total work space address in device for all groups.
  void* workSpaceData_;
  /// Size of total work space.
  size_t workSpaceInBytes_;

  /// Is or not select conv algorihtm.
  bool isSelectAlgo_;

public:
  explicit CudnnConvLayer(const LayerConfig& config) : ConvBaseLayer(config) {}

  ~CudnnConvLayer();

  /**
   * Intialization. Initialize member variables and create tenor descriptor.
   */
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  /**
   * Reshape is done each forward. Reshape tensor decriptor
   * inputDesc_, outputDesc_, convDesc_. And search the faster algo
   * or the fastest algo within a given memeory limit.
   */
  void reshape(int batchSize);
  void forward(PassType passType);
  void backward(const UpdateCallback& callback);
  void addBiases();
  void bpropBiases();
};

}  // namespace paddle
