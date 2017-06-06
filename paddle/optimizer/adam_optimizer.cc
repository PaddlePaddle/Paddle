#include "adam_optimizer.h"
#include <cmath>

namespace paddle {
namespace optimizer {

void AdamOptimizer::set_weight(Tensor *p) {
  parameter_ = p;
  size_t size = p->size();
  momentums_ = new Tensor(size);
  velocitys_ = new Tensor(size);
}

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
}  // namespace optimizer
}  // namespace paddle
