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

#include <cmath>

#include "adagrad_optimizer.h"

namespace paddle {
namespace optimizer {

void AdagradOptimizer::Update(const Tensor* gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  Tensor& param = *parameter_;
  Tensor& accum_g = *accum_gradient_;
  const Tensor& grad = *gradient;
  for (size_t i = 0; i < param.size(); ++i) {
    accum_g[i] += grad[i] * grad[i];
    param[i] += learning_rate * grad[i] / std::sqrt(accum_g[i] + epsilon_) +
                learning_rate * decay_ * param[i];
  }
}
std::string AdagradOptimizer::SerializeState() {
  AdagradOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*accum_gradient_, state.mutable_accum_gradient());
  return state.SerializeAsString();
}

void AdagradOptimizer::DeserializeState(const std::string& str) {
  AdagradOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());

  num_sample_passed_ = state.num_sample_passed();
  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.accum_gradient(), accum_gradient_);
}

}  // namespace optimizer
}  // namespace paddle
