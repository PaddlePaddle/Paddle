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

#pragma once

#include <vector>
#include "ConvBaseLayer.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief A subclass of ConvBaseLayer that is a superclass of both
 * ExpandConvLayer and ExpandConvTransLayer
 */
class ExpandConvBaseLayer : public ConvBaseLayer {
protected:
  /// For expand convolution.
  /// subM_ = numFilters_ / groups_.
  IntV subM_;
  /// subN_ = outputH_ * outputW_.
  IntV subN_;
  /// subK_ = channels_ * filterPixels_ * groups_.
  IntV subK_;

  /*The expandInput_ and transOutValue_ are used for CPU expand conv calc
   * Expand one sample at a time. shape:
   * (numChannels * filterPixels_, outputSizeH * outputSizeW)
   * */
  MatrixPtr expandInput_;
  /// The transpose of output, which is an auxiliary matrix.
  MatrixPtr transOutValue_;

public:
  explicit ExpandConvBaseLayer(const LayerConfig& config)
      : ConvBaseLayer(config) {}

  ~ExpandConvBaseLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  size_t getOutputSize();
  /**
   * Create or resize expandInput_.
   */
  void resetExpandInput(size_t height, size_t width);

  /**
   * Add shared bias.
   */
  void addSharedBias();

  /**
   * Add unshared bias.
   */
  void addUnsharedBias();
  /**
   * Expand one input sample.
   */
  void expandOneFrame(MatrixPtr image, size_t startIdx, int inIdx);

  /**
   * Expand one input sample and perform matrix multiplication.
   */
  void expandFwdOnce(MatrixPtr image, MatrixPtr out, int inIdx, int startIdx);

  void bpropSharedBias(MatrixPtr biases, MatrixPtr v);
  void bpropBiases(MatrixPtr v);
  void bpropWeights(MatrixPtr image, MatrixPtr out, int inpIdx);
  void bpropActs(MatrixPtr image, MatrixPtr out, int inpIdx);
};

}  // namespace paddle
