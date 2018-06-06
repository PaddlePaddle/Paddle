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
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * A base layer for SequenceLastInstanceLayer/AverageLayer/MaxLayer.
 *
 * Input: one or more sequences. Each sequence contains some instances.
 * If SequenceLevel = kNonSeq:
 *    Output: output size is the number of input sequences (NOT input instances)
 *    output[i] = seqlastin/average/max_{for each instance in this
 * sequence}{input[i]}
 *    If stride_ > 0:
 *        Check input sequence must not have sub-sequence
 *        Output: a shorten sequence. Stride is the step size by which we slide
 *                a window upon the input sequence, and the pooling operation
 *                is then applied to each interval independently.
 * If SequenceLevel = kSeq:
 *    Check input sequence must has sub-sequence
 *    Output: output size is the number of input sub-sequences
 *    output[i] = seqlastin/average/max_{for each instance in this
 * sub-sequence}{input[i]}
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
  int stride_;
  // Whether the input sequence is reversed or not.
  bool reversed_ = false;

 public:
  explicit SequencePoolLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
