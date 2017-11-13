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

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  AdadeltaOptimizer(
      Tensor *parameter, LrPolicy *lr, double rho, double epsilon, double decay)
      : ParameterOptimizer(parameter, lr),
        accum_gradient_(new Tensor(parameter->size())),
        accum_delta_(new Tensor(parameter->size())),
        update_delta_(new Tensor(parameter->size())),
        rho_(rho),
        epsilon_(epsilon),
        decay_(decay) {}

  ~AdadeltaOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
    if (accum_delta_) delete accum_delta_;
    if (update_delta_) delete update_delta_;
  }
  void Update(const Tensor *gradient);
  std::string SerializeState();
  void DeserializeState(const std::string &state);

private:
  Tensor *accum_gradient_;
  Tensor *accum_delta_;
  Tensor *update_delta_;
  double rho_;
  double epsilon_;
  double decay_;
};

}  // namespace optimizer
}  // namespace paddle
