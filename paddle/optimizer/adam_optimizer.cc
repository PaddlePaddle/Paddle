#include "adam_optimizer.h"


namespace paddle {
namespace optimizer {
template<class T>
AdamOptimizer<T>::AdamOptimizer(const ::paddle::OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  beta_1 = config.adam().beta_1();
  beta_2 = config.adam().beta_2();
  epsilon = config.adam().epsilon();
  decay = config.adam().decay();
}

template<class T>
void AdamOptimizer<T>::set_weight(const Tensor<T> *p) {
  size_t size = p->width();
  T* mptr = new T[size];
  momentums_ = Tensor<T>(mptr, size);
  T* vptr = new T[size];
  velocitys_ = Tensor<T>(vtpr, size);
}

template<class T>
void AdamOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate();
  for(size_t i=0; i<parameter_.size(); ++i) {
    accum_gradient[i] += gradient[i] * gradient[i];
    parameter_[i] += learning_rate * (gradient[i] / std::sqrt(accum_gradient[i] + epsilon) + decay * parameter_[i]);
  }
}


template class AdamOptimizer<float>;
template class AdamOptimizer<double>;
}
}
