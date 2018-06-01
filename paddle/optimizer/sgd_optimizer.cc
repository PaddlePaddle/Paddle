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

#include "sgd_optimizer.h"
#include "serialization.h"

namespace paddle {
namespace optimizer {

void SGDOptimizer::Update(const Tensor *gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  float velocity = 0.0;
  Tensor &param = *parameter_;
  const Tensor &grad = *gradient;
  Tensor &m = *momentums_;
  for (size_t i = 0; i < param.size(); ++i) {
    if (momentum_ == 0.0) {
      velocity = -learning_rate * grad[i] - learning_rate * decay_ * param[i];
    } else {
      m[i] = momentum_ * m[i] - learning_rate * grad[i] -
             learning_rate * decay_ * param[i];
      velocity = m[i];
    }
    if (nesterov_) {
      param[i] += momentum_ * velocity - learning_rate * grad[i];
    } else {
      param[i] += velocity;
    }
  }
}

std::string SGDOptimizer::SerializeState() {
  SGDOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);
  TensorToProto(*parameter_, state.mutable_parameter());
  if (momentum_ != 0.0) TensorToProto(*momentums_, state.mutable_momentums());
  return state.SerializeAsString();
}

void SGDOptimizer::DeserializeState(const std::string &str) {
  SGDOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());
  num_sample_passed_ = state.num_sample_passed();
  ProtoToTensor(state.parameter(), parameter_);
  if (momentum_ != 0.0) ProtoToTensor(state.momentums(), momentums_);
}

}  // namespace optimizer
}  // namespace paddle
