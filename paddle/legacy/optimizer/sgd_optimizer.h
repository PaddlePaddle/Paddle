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

class SGDOptimizer : public ParameterOptimizer {
 public:
  SGDOptimizer(Tensor* parameter, LrPolicy* lr, double m, double d, bool n)
      : ParameterOptimizer(parameter, lr),
        momentums_(nullptr),
        momentum_(m),
        decay_(d),
        nesterov_(n) {
    if (momentum_ != 0.0) {
      size_t size = parameter->size();
      momentums_ = new Tensor(size);
    }
  }
  virtual ~SGDOptimizer() {
    if (momentums_) delete momentums_;
  }
  void Update(const Tensor* gradient);
  std::string SerializeState();
  void DeserializeState(const std::string& state);

 private:
  Tensor* momentums_;
  double momentum_;
  double decay_;
  bool nesterov_;
};

}  // namespace optimizer
}  // namespace paddle
