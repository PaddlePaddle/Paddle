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

#include "Layer.h"
#include "paddle/math/MathUtils.h"
namespace paddle {

/**
 * @brief A Base Convolution Layer, which convolves the input image
 * with learned filters and (optionally) adds biases.
 */

class ConvBaseLayer : public Layer {
 protected:
  typedef std::vector<int> IntV;

  /// True if it's deconv layer, false if it's convolution layer
  bool isDeconv_;

  /// The number of filters.
  int numFilters_;
  /// The x dimension of the padding.
  IntV padding_;
  /// The y dimension of the padding.
  IntV paddingY_;
  /// The x dimension of the stride.
  IntV stride_;
  /// The y dimension of the stride.
  IntV strideY_;
  /// The x dimension of the dilation.
  IntV dilation_;
  /// The y dimension of the dilation.
  IntV dilationY_;
  /// The x dimension of a filter kernel.
  IntV filterSize_;
  /// The y dimension of a filter kernel.
  IntV filterSizeY_;
  /// The spatial dimensions of the convolution input.
  IntV channels_;
  /// The spatial dimensions of input feature map height.
  IntV imgSizeH_;
  /// The spatial dimensions of input feature map width.
  IntV imgSizeW_;
  /// filterPixels_ = filterSizeX_ * filterSizeY_.
  IntV filterPixels_;
  /// filterChannels_ = channels_/groups_.
  IntV filterChannels_;
  /// The spatial dimensions of output feature map height.
  IntV outputH_;
  /// The spatial dimensions of output feature map width.
  IntV outputW_;

  IntV outputD_;
  IntV imgSizeD_;
  IntV filterSizeZ_;
  IntV strideZ_;
  IntV paddingZ_;

  /// Group size, refer to grouped convolution in
  /// Alex Krizhevsky's paper: when group=2, the first half of the
  /// filters are only connected to the first half of the input channels,
  /// and the second half only connected to the second half.
  IntV groups_;
  /// Whether the bias is shared for feature in each channel.
  bool sharedBiases_;

  /// shape of weight: (numChannels * filterPixels_, numFilters)
  WeightList weights_;
  /// If shared_biases is false shape of bias: (numFilters_, 1)
  /// If shared_biases is ture shape of bias:
  /// (numFilters_ * outputX * outputY, 1)
  std::unique_ptr<Weight> biases_;

  /// True by default. The only difference is the calculation
  /// of output size.
  bool caffeMode_;

 public:
  explicit ConvBaseLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  /**
   * imgSizeH_ and imgSizeW_ will be set according to the previous input layers
   * in this function. Then it will calculate outputH_ and outputW_ and set them
   * into output argument.
   */
  virtual size_t calOutputSize();

  Weight& getWeight(int idx) { return *weights_[idx]; }
};

}  // namespace paddle
