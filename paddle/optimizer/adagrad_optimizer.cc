#include "adagrad_optimizer.h"

namespace paddle {
namespace optimizer {
template<class T>
AdagradOptimizer<T>::AdagradOptimizer(const ::paddle::OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  epsilon = config.adagrad().epsilon();
  decay = config.adagrad().decay();
}

template<class T>
void AdagradOptimizer<T>::set_weight(const Tensor<T> *p) {
  size_t size = p->width();
  T* gptr = new T[size];
  accum_gradient = Tensor<T>(gptr, size);
  T* dptr = new T[size];
  accum_delta = Tensor<T>(dtpr, size);
  T* dptr_current = new T[size];
  update_delta = Tensor<T>(dptr_current, size);
}

template<class T>
void AdagradOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate();
  for(size_t i=0; i<parameter_.size(); ++i) {
    accum_gradient[i] += gradient[i] * gradient[i];
    parameter_[i] += learning_rate * (gradient[i] / std::sqrt(accum_gradient[i] + epsilon) + decay * parameter_[i]);
  }
}


template class AdagradOptimizer<float>;
template class AdagradOptimizer<double>;
}
}
