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

#include "ConvTransBaseLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>

namespace paddle {

/**
 * @brief A subclass of convolution layer.
 * This layer expands input and use matrix multiplication to
 * calculate convolution operation.
 *
 * The config file api is img_conv_layer.
 */
class ExpandConvTransLayer : public ConvTransBaseLayer {
protected:
  /// For expand convolution.
  /// subM_ = numFilters_ / groups_.
  IntV subM_;
  /// subN_ = outputH_ * outputW_.
  IntV subN_;
  /// subK_ = channels_ * filterPixels_ * groups_.
  IntV subK_;
  /// The spatial dimensions of height of input feature map.
  IntV imgSizeH_;
  /// The spatial dimensions of width of input feature map.
  IntV imgSizeW_;
  /// The spatial dimensions of height of output feature map.
  IntV outputH_;
  /// The spatial dimensions of width of output feature map.
  IntV outputW_;
  /// Expand one sample at a time. shape:
  /// (numChannels * filterPixels_, outputSizeH * outputSizeW)
  MatrixPtr expandInput_;
  /// The transpose of output, which is an auxiliary matrix.
  MatrixPtr transOutValue_;

public:
  explicit ExpandConvTransLayer(const LayerConfig& config) :
      ConvTransBaseLayer(config) {}

  ~ExpandConvTransLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  size_t getSize();

  /**
   * Create or resize expandInput_.
   */
  void resetExpandInput(size_t height, size_t width);

  /**
   * Create or resize transOutValue_.
   */
  void resetConvOutput(size_t batchSize, int inIdx);

  /**
   * Expand one input sample.
   */
  void expandOneFrame(MatrixPtr image, size_t startIdx, int inIdx);

  /**
   * Expand one output image and perform matrix multiplication.
   */
  void expandBackOnce(MatrixPtr image, int inIdx, int startIdx);

  /**
   * Perform matrix multiplication on one output and then shrink.
   */
  void shrinkFwd(MatrixPtr output, int inpIdx);

  /**
   * Add shared bias.
   */
  void addSharedBias();

  /**
   * Add unshared bias.
   */
  void addUnsharedBias();
  void forward(PassType passType);
  void bpropSharedBias(MatrixPtr biases, MatrixPtr v);
  void bpropBiases(MatrixPtr v);
  void backward(const UpdateCallback& callback);
  void bpropWeights(MatrixPtr v, int inpIdx);
  void bpropActs(MatrixPtr v, int inpIdx);
};

}  // namespace paddle
