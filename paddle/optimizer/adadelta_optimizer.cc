#include "adadelta_optimizer.h"
#include <algorithm>
#include <cmath>

namespace paddle {
namespace optimizer {

void AdadeltaOptimizer::set_weight(Tensor* p) {
  parameter_ = p;
  size_t size = p->size();
  real* gptr = new real[size];
  accum_gradient_ = new Tensor(gptr, size);
  real* dptr = new real[size];
  accum_delta_ = new Tensor(dptr, size);
  real* dptr_current = new real[size];
  update_delta_ = new Tensor(dptr_current, size);
}

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
}  // namespace optimizer
}  // namespace paddle
