#include "adadelta_optimizer.h"
#include <algorithm>
#include <cmath>

namespace paddle {
namespace optimizer {

void AdadeltaOptimizer::set_weight(Tensor* p) {
  size_t size = p->size();
  real* gptr = new real[size];
  accum_gradient = new Tensor(gptr, size);
  real* dptr = new real[size];
  accum_delta = new Tensor(dptr, size);
  real* dptr_current = new real[size];
  update_delta = new Tensor(dptr_current, size);
}

void AdadeltaOptimizer::update(const Tensor* gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  Tensor& param = *parameter_;
  const Tensor& grad = *gradient;
  Tensor& accum_g = *accum_gradient;
  Tensor& accum_d = *accum_delta;
  Tensor& update_d = *update_delta;
  for (size_t i = 0; i < param.size(); ++i) {
    accum_g[i] = rho * accum_g[i] + (1.0 - rho) * grad[i] * grad[i];

    update_d[i] = std::sqrt(accum_d[i] + epsilon) /
                  std::sqrt(accum_g[i] + epsilon) * grad[i];

    accum_d[i] = rho * accum_d[i] + (1.0 - rho) * update_d[i] * update_d[i];

    param[i] -= learning_rate * update_d[i] + learning_rate * decay * param[i];
  }
}
}  // namespace optimizer
}  // namespace paddle
