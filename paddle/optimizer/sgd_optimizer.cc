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

const char *SGDOptimizer::SerializeState(int *state_len) {
  SGDOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  TensorToProto(*parameter_, state.mutable_parameter());
  if (momentum_ != 0.0) TensorToProto(*momentums_, state.mutable_momentums());
  auto str = state.SerializeAsString();
  *state_len = str.size();
  return str.c_str();
}

void SGDOptimizer::DeserializeState(const std::string &str) {
  SGDOptimizerState state;
  state.ParseFromString(str);
  num_sample_passed_ = state.num_sample_passed();
  ProtoToTensor(state.parameter(), parameter_);
  if (momentum_ != 0.0) ProtoToTensor(state.parameter(), momentums_);
}

}  // namespace optimizer
}  // namespace paddle
