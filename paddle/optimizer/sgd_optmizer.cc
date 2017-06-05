#include "sgd_optimizer.h"

namespace paddle {
namespace optimizer {

void SGDOptimizer::set_weight(Tensor *p) {
  //  ParameterOptimizer::set_weight(p);
  size_t size = p->size();
  // TODO: fix it with align aware allocator bind to Tensor
  if (momentum != 0.0) {
    real *ptr = new real[size];
    momentums_ = new Tensor(ptr, size);
  }
}

void SGDOptimizer::update(const Tensor &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  real velocity = 0.0;
  Tensor &param = *parameter_;
  for (size_t i = 0; i < parameter_->size(); ++i) {
    if (momentum == 0.0) {
      velocity =
          -learning_rate * gradient[i] - learning_rate * decay * parameter_[i];
    } else {
      momentums_[i] = momentum * momentums_[i] - learning_rate * gradient[i] -
                      learning_rate * decay * parameter_[i];
      velocity = momentums_[i];
    }
    if (nesterov) {
      parameter_[i] += momentum * velocity - learning_rate * gradient[i];
    } else {
      parameter_[i] += velocity;
    }
  }
}

}  // namespace optimizer
}  // namespace paddle
