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

class AdamOptimizer : public ParameterOptimizer {
 public:
  AdamOptimizer(Tensor *parameter,
                LrPolicy *lr,
                double beta_1,
                double beta_2,
                double epsilon,
                double decay)
      : ParameterOptimizer(parameter, lr),
        momentums_(new Tensor(parameter->size())),
        velocitys_(new Tensor(parameter->size())),
        beta_1_(beta_1),
        beta_2_(beta_2),
        epsilon_(epsilon),
        decay_(decay) {}
  ~AdamOptimizer() {
    if (momentums_) delete momentums_;
    if (velocitys_) delete velocitys_;
  }
  void Update(const Tensor *gradient);
  std::string SerializeState();
  void DeserializeState(const std::string &state);

 private:
  Tensor *momentums_;
  Tensor *velocitys_;
  double beta_1_;
  double beta_2_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
