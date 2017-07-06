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

const char* AdadeltaOptimizer::SerializeState(int* state_len) {
  AdadeltaOptimizerState state;
  // TODO(zhihong) : add lr_policy serialization
  state.set_num_sample_passed(num_sample_passed_);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*accum_gradient_, state.mutable_accum_gradient());
  TensorToProto(*accum_delta_, state.mutable_accum_delta());
  TensorToProto(*update_delta_, state.mutable_update_delta());
  auto str = state.SerializeAsString();
  *state_len = str.size();
  return str.c_str();
}

void AdadeltaOptimizer::DeserializeState(const std::string& str) {
  AdadeltaOptimizerState state;
  state.ParseFromString(str);
  // TODO(zhihong) : add lr_policy DeserializeState
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.accum_gradient(), accum_gradient_);
  ProtoToTensor(state.accum_delta(), accum_delta_);
  ProtoToTensor(state.update_delta(), update_delta_);
}

}  // namespace optimizer
}  // namespace paddle
