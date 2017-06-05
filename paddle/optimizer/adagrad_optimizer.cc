#include "adagrad_optimizer.h"

namespace paddle {
namespace optimizer {

void AdagradOptimizer::set_weight(Tensor* p) {
  size_t size = p->width();
  real* gptr = new real[size];
  accum_gradient = Tensor(gptr, size);
  real* dptr = new real[size];
  accum_delta = Tensor(dtpr, size);
  real* dptr_current = new real[size];
  update_delta = Tensor(dptr_current, size);
}

void AdagradOptimizer::update(const Tensor& gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate();
  for (size_t i = 0; i < parameter_.size(); ++i) {
    accum_gradient[i] += gradient[i] * gradient[i];
    parameter_[i] +=
        learning_rate * (gradient[i] / std::sqrt(accum_gradient[i] + epsilon) +
                         decay * parameter_[i]);
  }
}

}  // namespace optimizer
}  // namespace paddle
