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
#include <string>

namespace paddle {

namespace enumeration_wrapper {
enum PassType {
  PASS_TRAIN,   // Train pass
  PASS_TEST,    // Test pass
  PASS_GC,      // Gradient Check pass
  PASS_METRIC,  // pass for generate template output with no drop rate.
};

enum ParameterType {
  PARAMETER_VALUE = 0,
  PARAMETER_GRADIENT,
  PARAMETER_MOMENTUM,

  // Used by ParameterAverager
  PARAMETER_SUM1,
  PARAMETER_SUM2,
  PARAMETER_SUM3,

  //   also used by AdagradParameterUpdater/AdadeltaParameterUpdater
  PARAMETER_LEARNING_RATE,

  // Used by Sparse SGD update
  PARAMETER_UPDATE_TIME,

  // Used by async_sgd
  // Change of the parameter since last remote update
  PARAMETER_DELTA,

  // Used by BatchRemoteParameterUpdater
  PARAMETER_GRADIENT_SUM,

  // Used by AdagradParameterUpdater/AdadeltaParameterUpdater
  PARAMETER_GRADIENT_SQURESUM,
  PARAMETER_GRADIENT_SQURESUM1,

  // Used by SparseConnected layer
  PARAMETER_ROWS,
  PARAMETER_COLS,

  // Used by Adam Optimizer.
  PARAMETER_SECOND_MOMENTUM,

  // Used By AdaMax Optimizer.
  PARAMETER_WEIGHTED_INFINITY_NORM,

  // Used by remote parameter average
  PARAMETER_APPLY,

  // Used by sparse momentum
  PARAMETER_MOMENTUM_UT,
  PARAMETER_MOMENTUM_VT,

  NUM_PARAMETER_TYPES,
};

}  // namespace enumeration_wrapper

//! explicit import enum into paddle namespace.
using namespace enumeration_wrapper;  // NOLINT

class TrainAlgorithm {
public:
  static const std::string SGD;
  static const std::string AsyncSGD;
  static const std::string OWLQN;

  static inline bool isValid(const std::string& algo) {
    return algo == SGD || algo == AsyncSGD || algo == OWLQN;
  }
};

#ifdef __AVX__
const int ALIGN_HINT = 32;
#else
const int ALIGN_HINT = 16;
#endif

}  // namespace paddle
