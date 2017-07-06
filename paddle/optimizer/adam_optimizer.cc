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

const char *AdamOptimizer::SerializeState(int *state_len) {
  AdamOptimizerState state;
  // TODO(zhihong) : add lr_policy serialization
  state.set_num_sample_passed(num_sample_passed_);
  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*momentums_, state.mutable_momentums());
  TensorToProto(*velocitys_, state.mutable_velocitys());
  auto str = state.SerializeAsString();
  *state_len = str.size();
  return str.c_str();
}

void AdamOptimizer::DeserializeState(const std::string &str) {
  AdamOptimizerState state;
  state.ParseFromString(str);
  // TODO(zhihong) : add lr_policy DeserializeState
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.momentums(), momentums_);
  ProtoToTensor(state.velocitys(), velocitys_);
}
}  // namespace optimizer
}  // namespace paddle
