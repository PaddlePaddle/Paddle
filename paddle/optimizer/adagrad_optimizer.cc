#include <cmath>

#include "adagrad_optimizer.h"

namespace paddle {
namespace optimizer {

void AdagradOptimizer::set_weight(Tensor* p) {
  size_t size = p->size();
  real* gptr = new real[size];
  accum_gradient = new Tensor(gptr, size);
}

void AdagradOptimizer::update(const Tensor* gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  Tensor& param = *parameter_;
  const Tensor& grad = *gradient;
  Tensor& accum_g = *accum_gradient;
  for (size_t i = 0; i < param.size(); ++i) {
    accum_g[i] += grad[i] * grad[i];
    param[i] += learning_rate * grad[i] / std::sqrt(accum_g[i] + epsilon) +
                learning_rate * decay * param[i];
  }
}

}  // namespace optimizer
}  // namespace paddle
