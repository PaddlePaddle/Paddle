#include "serialization.h"
#include "sgd_optimizer.h"

namespace paddle {
namespace optimizer {

void SGDOptimizer::set_weight(Tensor *p) {
  parameter_ = p;
  size_t size = p->size();
  // TODO: fix it with align aware allocator bind to Tensor
  if (momentum_ != 0.0) {
    real *ptr = new real[size];
    momentums_ = new Tensor(ptr, size);
  }
}

void SGDOptimizer::Update(const Tensor *gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  real velocity = 0.0;
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

const char *SGDOptimizer::SerializeState() {
  OptimizerState state;
  // version is a global const value
  state.set_version(kOptimizerVersion);
  TensorToProto(*parameter_, state.add_data());
  TensorToProto(*momentums_, state.add_data());
  state.add_hyperparam(momentum_);
  return state.SerializeAsString().c_str();
}

void SGDOptimizer::DeSerializeState(const std::string &str) {
  OptimizerState state;
  state.ParseFromString(str);
  CHECK(state.version() == kOptimizerVersion)
      << "error version of state"
      << "expected : " << kOptimizerVersion << "get : " << state.version();

  ProtoToTensor(state.data(0), parameter_);
  if (state.data_size() == 2) {
    ProtoToTensor(state.data(1), momentums_);
    momentum_ = state.hyperparam(0);
  }
}

}  // namespace optimizer
}  // namespace paddle
