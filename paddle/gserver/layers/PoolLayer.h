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
#include "paddle/math/Matrix.h"
#include <vector>

namespace paddle {

/**
 * @brief Basic parent layer of pooling
 * Pools the input within regions
 */
class PoolLayer : public Layer {
protected:
  size_t channels_, sizeX_, stride_, outputX_, imgSize_;
  int confPadding_;

  size_t sizeY_;
  size_t imgSizeY_;
  size_t strideY_;
  size_t outputY_;
  int confPaddingY_;

  std::string poolType_;

public:
  explicit PoolLayer(const LayerConfig& config) : Layer(config) {}

  /**
   * @brief create pooling layer by pool_type
   */
  static Layer* create(const LayerConfig& config);

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  /**
   * Calculate output size according window size and padding size.
   */
  int outputSize(int imageSize, int windowSize, int padding, int stride) {
    int outputSize;
    outputSize =
        (imageSize - windowSize + 2 * padding + stride - 1) / stride + 1;
    return outputSize;
  }
};

/**
 * A base layer for SequenceLastInstanceLayer/AverageLayer/MaxLayer.
 *
 * Input: one or more sequences. Each sequence contains some instances.
 * If SequenceLevel = kNonSeq:
 *    Output: output size is the number of input sequences (NOT input instances)
 *    output[i] = seqlastin/average/max_{for each instance in this sequence}{input[i]}
 * If SequenceLevel = kSeq:
 *    Check input sequence must has sub-sequence
 *    Output: output size is the number of input sub-sequences
 *    output[i] = seqlastin/average/max_{for each instance in this sub-sequence}{input[i]}
 *
 * The config file api is pooling_layer.
 */
class SequencePoolLayer : public Layer {
protected:
  int type_;
  std::unique_ptr<Weight> biases_;
  enum SequenceLevel { kNonSeq = 0, kSeq = 1 };
  size_t newBatchSize_;
  ICpuGpuVectorPtr startPositions_;

public:
  explicit SequencePoolLayer(const LayerConfig& config) : Layer(config) {}

  ~SequencePoolLayer() {}

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

}  // namespace paddle
