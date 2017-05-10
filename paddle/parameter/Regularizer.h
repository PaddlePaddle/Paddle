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

#include "ParameterUpdaterBase.h"

namespace paddle {

// Regularizer function for parameter, e.g. L1/L2
class Regularizer {
public:
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,  // learningrate from optimizer
                      int t0,             // last occurence time
                      int t) const = 0;   // current time
  virtual ~Regularizer() {}

  static Regularizer* get(const std::vector<ParameterType>& types,
                          const ParameterConfig& paraConfig);
};

// L1 Regularizer, |w|_1
class L1Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
  }
};

// L1 Lr Regularizer
class L1LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
  }
};

// L2 Regularizer, |w|_2^2
class L2Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL2(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L2 Lr Regularizer
class L2LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL2(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L1 + L2 Regularizer, |w|_1 + |w|_2^2
class L1L2Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
    vecs[PARAMETER_VALUE]->applyL2(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L1 + L2 Lr Regularizer
class L1L2LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
    vecs[PARAMETER_VALUE]->applyL2(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

}  // namespace paddle
