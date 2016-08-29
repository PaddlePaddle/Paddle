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

namespace paddle {

/**
 * A layer for "internal average" for sequence input.
 * Input: one or more sequences. Each sequence contains some instances.
 * If AverageLevel = kNonSeq:
 *    Output: output size is the number of input sequences (NOT input instances)
 *    output[i] = average_{for each instance in this sequence}{input[i]}
 * If AverageLevel = kSeq:
 *    Check input sequence must has sub-sequence
 *    Output: output size is the number of input sub-sequences
 *    output[i] = average_{for each instance in this sub-sequence}{input[i]}
 */

class AverageLayer : public Layer {
public:
  enum AverageStrategy { kAverage = 0, kSum = 1, kAverageSquareRootN = 2 };
  enum AverageLevel { kNonSeq = 0, kSeq = 1 };
  explicit AverageLayer(const LayerConfig& config) : Layer(config) {}

  ~AverageLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

protected:
  std::unique_ptr<Weight> biases_;
  MatrixPtr outMtx_;
  MatrixPtr dataMtx_;
  int mode_;
  int type_;
};

}  // namespace paddle
