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
 * A layer for "Expand Dense data or (sequence data where the length of each
 * sequence is one) to sequence data."
 *
 * It should have exactly 2 input, one for data, one for size:
 * - first one for data
 *   - If ExpandLevel = kNonSeq: dense data
 *   - If ExpandLevel = kSeq: sequence data where the length of each sequence is
 * one
 * - second one only for sequence info
 *   - should be sequence data with or without sub-sequence.
 *
 * And the output size is the batch size(not instances) of second input.
 *
 * The config file api is expand_layer.
 */

class ExpandLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;
  /// if input[0] is dense data, ExpandLevel=kNonSeq;
  /// if input[0] is sequence data, ExpandLevel=kSeq
  enum ExpandLevel { kNonSeq = 0, kSeq = 1 };
  /// store the ExpandLevel
  int type_;
  /// expanded sequenceStartPositions or subSequenceStartPositions
  /// of input[1]
  ICpuGpuVectorPtr expandStartsPos_;

 public:
  explicit ExpandLayer(const LayerConfig& config) : Layer(config) {}

  ~ExpandLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
