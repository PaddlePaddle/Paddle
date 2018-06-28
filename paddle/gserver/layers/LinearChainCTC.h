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

#include <vector>
#include "paddle/math/Matrix.h"

namespace paddle {

class LinearChainCTC {
 public:
  LinearChainCTC(int numClasses, bool normByTimes);

  // Calculate the negative log probability as loss
  real forward(real* softmaxSeq,
               int softmaxSeqLen,
               int* labelSeq,
               int labelSeqLen);

  // calculate the gradient
  void backward(real* softmaxSeq,
                real* softmaxSeqGrad,
                int* labelSeq,
                int labelSeqLen);

 protected:
  int numClasses_, blank_, totalSegments_, totalTime_;
  bool normByTimes_;
  bool isInvalid_;

  MatrixPtr logActs_, forwardVars_, backwardVars_, gradTerms_;

  real logProb_;

  void segmentRange(int& start, int& end, int time);
};

}  // namespace paddle
