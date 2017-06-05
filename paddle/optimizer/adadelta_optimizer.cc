#include "adadelta_optimizer.h"
#include <algorithm>

namespace paddle {
namespace optimizer {

void AdadeltaOptimizer::set_weight(Tensor* p) {
  size_t size = p->size();
  real* gptr = new real[size];
  accum_gradient = Tensor(gptr, size);
  real* dptr = new real[size];
  accum_delta = Tensor(dptr, size);
  real* dptr_current = new real[size];
  update_delta = Tensor(dptr_current, size);
}

void AdadeltaOptimizer::update(const Tensor& gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  for (size_t i = 0; i < parameter_->size(); ++i) {
    accum_gradient[i] =
        rho * accum_gradient[i] + (1.0 - rho) * gradient[i] * gradient[i];

    update_delta[i] = std::sqrt(accum_delta[i] + epsilon) /
                      std::sqrt(accum_gradient[i] + epsilon) * gradient[i];

    accum_delta[i] =
        rho * accum_delta[i] + (1.0 - rho) * update_delta[i] * update_delta[i];

    parameter_[i] -=
        learning_rate * update_delta[i] + learning_rate * decay * parameter_[i];
  }
}
}  // namespace optimizer
}  // namespace paddle
