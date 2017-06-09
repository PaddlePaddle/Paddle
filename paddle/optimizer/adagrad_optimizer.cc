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
const char* SGDOptimizer::SerializeState(int* state_len) { NIMPL; }

void SGDOptimizer::DeSerializeState(const std::string& str) { NIMPL; }
// namespace optimizer
}  // namespace optimizer
