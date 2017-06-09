#include "adam_optimizer.h"
#include <cmath>

namespace paddle {
namespace optimizer {

void AdamOptimizer::Update(const Tensor *gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  double coef1 = 1.0 - std::pow(beta_1_, num_sample_passed_);
  double coef2 = 1.0 - std::pow(beta_2_, num_sample_passed_);
  learning_rate *= std::sqrt(coef2) / coef1;
  Tensor &param = *parameter_;
  const Tensor &grad = *gradient;
  Tensor &m = *momentums_;
  Tensor &v = *velocitys_;
  for (size_t i = 0; i < param.size(); ++i) {
    m[i] = beta_1_ * m[i] + (1.0 - beta_1_) * grad[i];
    v[i] = beta_2_ * v[i] + (1.0 - beta_2_) * grad[i] * grad[i];
    param[i] -=
        learning_rate * (m[i] / std::sqrt(v[i] + epsilon_) + decay_ * param[i]);
  }
}

const char *AdadeltaOptimizer::SerializeState(int *state_len) {
  OptimizerState state;
  state.set_learning_rate(lr_policy_->LearningRate(num_sample_passed_));
  state.set_num_sample_passed(num_sample_passed_);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*velocitys_, state.mutable_momentums());

  state.set_beta_1(beta_1_);
  state.set_beta_2(beta_2_);
  state.set_decay(decay_);
  *state_len += CalStateSize(
      parameter_, momentums_, velocitys_, beta_1_, beta_2, epsilon_ decay_);
  return state.SerializeAsString().c_str();
}

void AdadeltaOptimizer::DeSerializeState(const std::string &str) {
  OptimizerState state;
  state.ParseFromString(str);
  lr_policy_->set(state.learning_rate());
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.velocitys(), velocitys__);
  beta_1_ = state.beta_1();
  beta_2_ = state.beta_2();
}
}  // namespace optimizer
}  // namespace paddle
