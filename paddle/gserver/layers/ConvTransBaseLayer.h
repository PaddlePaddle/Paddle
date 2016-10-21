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

#include "Layer.h"
namespace paddle {

/**
 * @brief A Base Convolution Layer, which convolves the input image
 * with learned filters and (optionally) adds biases.
 */

class ConvTransBaseLayer : public Layer {
protected:
  typedef std::vector<int> IntV;

  /// The number of channel in image (the output of the deconv layer).
  int channel_;
  /// The x dimension of the padding.
  IntV padding_;
  /// The y dimension of the padding.
  IntV paddingY_;
  /// The x dimension of the stride.
  IntV stride_;
  /// The y dimension of the stride.
  IntV strideY_;
  /// The x dimension of a filter kernel.
  IntV filterSize_;
  /// The y dimension of a filter kernel.
  IntV filterSizeY_;
  /// The number of filters(i.e. the number channels of the deconv layer input)
  IntV numFilters_;
  /// The spatial dimensions of input feature map.
  IntV imgSize_;
  /// The total pixel size of input feature map.
  /// imgPixels_ = imgSizeX_ * imgSizeY_.
  IntV imgPixels_;
  /// filterPixels_ = filterSizeX_ * filterSizeY_.
  IntV filterPixels_;
  /// filterChannels_ = channels_/groups_.
  IntV filterChannels_;
  /// The spatial dimensions of output feature map.
  IntV outputX_;
  /// The spatial dimensions of output feature map.
  IntV outputs_;
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
  explicit ConvTransBaseLayer(const LayerConfig& config) : Layer(config) {}

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  Weight& getWeight(int idx) { return *weights_[idx]; }

  /**
   * Calculate image size based on caffeMode_ from outputSize.
   * - input(+padding): 0123456789
   * - imageSize(+padding) = 10;
   * - filterSize = 3;
   * - stride = 2;
   * - caffeMode_ is true:
       - output: (012), (234), (456), (678)
       - outputSize = 4;
   * - caffeMode_ is false:
   *   - output: (012), (234), (456), (678), (9)
   *   - outputSize = 5;
   */

  int imageSize(int outputSize, int filterSize, int padding, int stride) {
    int imageSize;
    if (!caffeMode_) {
     imageSize =
         (outputSize - 1) * stride + filterSize - 2 * padding - stride + 1;
    } else {
     imageSize = (outputSize - 1) * stride + filterSize - 2 * padding;
    }
    CHECK_GE(imageSize, 1);
    return imageSize;
  }
};

}  // namespace paddle
