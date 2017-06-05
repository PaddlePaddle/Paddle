#include "adam_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
void AdamOptimizer<T>::set_weight(const Tensor<T> *p) {
  size_t size = p->width();
  T *mptr = new T[size];
  momentums_ = Tensor<T>(mptr, size);
  T *vptr = new T[size];
  velocitys_ = Tensor<T>(vtpr, size);
}

template <class T>
void AdamOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  double coef1 = 1.0 - std::pow(beta_1, num_sample_passed);
  double coef2 = 1.0 - std::pow(beta_2, num_sample_passed);
  learning_rate *= std::sqrt(coef2) / coef1;
  for (size_t i = 0; i < parameter_->size(); ++i) {
    momentums_[i] = beta_1 * momentums_[i] + (1.0 - beta_1) * gradient[i];
    velocitys_[i] =
        beta_2 * velocitys_[i] + (1.0 - beta_2) * gradient[i] * gradient[i];
    parameter_[i] -=
        learning_rate * (momentums_[i] / std::sqrt(velocitys_[i] + epsilon) +
                         decay * parameter_[i]);
  }
}

template class AdamOptimizer<float>;
template class AdamOptimizer<double>;
}  // namespace optimizer
}  // namespace paddle
