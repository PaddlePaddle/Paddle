#include "serialization.h"
#include "sgd_optimizer.h"

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
  OptimizerState state;
  state.set_learning_rate(lr_policy_->LearningRate(num_sample_passed_));
  state.set_num_sample_passed(num_sample_passed_);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*momentums_, state.mutable_momentums());
  state.set_momentum(momentum_);
  state.set_decay(decay_);
  state.set_nesterov(nesterov_);
  *state_len +=
      CalStateSize(parameter_, momentums_, momentum_, decay_, nesterov_);
  return state.SerializeAsString().c_str();
}

void SGDOptimizer::DeSerializeState(const std::string &str) {
  OptimizerState state;
  state.ParseFromString(str);
  lr_policy_->set(state.learning_rate());
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.parameter(), momentums_);
  momentum_ = state.momentum();
}

}  // namespace optimizer
}  // namespace paddle
