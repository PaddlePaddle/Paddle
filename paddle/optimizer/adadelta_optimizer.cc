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

#include "adadelta_optimizer.h"
#include <algorithm>
#include <cmath>

namespace paddle {
namespace optimizer {

void AdadeltaOptimizer::Update(const Tensor* gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  Tensor& param = *parameter_;
  const Tensor& grad = *gradient;
  Tensor& accum_g = *accum_gradient_;
  Tensor& accum_d = *accum_delta_;
  Tensor& update_d = *update_delta_;
  for (size_t i = 0; i < param.size(); ++i) {
    accum_g[i] = rho_ * accum_g[i] + (1.0 - rho_) * grad[i] * grad[i];

    update_d[i] = std::sqrt(accum_d[i] + epsilon_) /
                  std::sqrt(accum_g[i] + epsilon_) * grad[i];

    accum_d[i] = rho_ * accum_d[i] + (1.0 - rho_) * update_d[i] * update_d[i];

    param[i] -= learning_rate * update_d[i] + learning_rate * decay_ * param[i];
  }
}

std::string AdadeltaOptimizer::SerializeState() {
  AdadeltaOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*accum_gradient_, state.mutable_accum_gradient());
  TensorToProto(*accum_delta_, state.mutable_accum_delta());
  TensorToProto(*update_delta_, state.mutable_update_delta());
  return state.SerializeAsString();
}

void AdadeltaOptimizer::DeserializeState(const std::string& str) {
  AdadeltaOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.accum_gradient(), accum_gradient_);
  ProtoToTensor(state.accum_delta(), accum_delta_);
  ProtoToTensor(state.update_delta(), update_delta_);
}

}  // namespace optimizer
}  // namespace paddle
