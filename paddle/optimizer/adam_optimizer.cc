#include "adam_optimizer.h"
#include <cmath>

namespace paddle {
namespace optimizer {

void AdamOptimizer::set_weight(Tensor *p) {
  size_t size = p->size();
  real *mptr = new real[size];
  momentums_ = new Tensor(mptr, size);
  real *vptr = new real[size];
  velocitys_ = new Tensor(vptr, size);
}

void AdamOptimizer::update(const Tensor *gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  double coef1 = 1.0 - std::pow(beta_1, num_sample_passed);
  double coef2 = 1.0 - std::pow(beta_2, num_sample_passed);
  learning_rate *= std::sqrt(coef2) / coef1;
  Tensor &param = *parameter_;
  const Tensor &grad = *gradient;
  Tensor &m = *momentums_;
  Tensor &v = *velocitys_;
  for (size_t i = 0; i < param.size(); ++i) {
    m[i] = beta_1 * m[i] + (1.0 - beta_1) * grad[i];
    v[i] = beta_2 * v[i] + (1.0 - beta_2) * grad[i] * grad[i];
    param[i] -=
        learning_rate * (m[i] / std::sqrt(v[i] + epsilon) + decay * param[i]);
  }
}
}  // namespace optimizer
}  // namespace paddle
