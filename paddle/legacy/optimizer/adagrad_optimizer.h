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

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdagradOptimizer : public ParameterOptimizer {
 public:
  AdagradOptimizer(Tensor *parameter,
                   LrPolicy *lr,
                   double epsilon,
                   double decay)
      : ParameterOptimizer(parameter, lr),
        accum_gradient_(new Tensor(parameter->size())),
        epsilon_(epsilon),
        decay_(decay) {}
  ~AdagradOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
  }
  void Update(const Tensor *gradient);
  std::string SerializeState();
  void DeserializeState(const std::string &state);

 private:
  Tensor *accum_gradient_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
