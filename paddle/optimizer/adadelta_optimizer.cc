#include "adadelta_optimizer.h"
#include <algorithm>

namespace paddle {
namespace optimizer {
template<class T>
AdadeltaOptimizer<T>::AdadeltaOptimizer(const ::paddle::OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  rho = config.adadelta().rho();
  epsilon = config.adadelta().epsilon();
  decay = config.adadelta().decay();
}

template<class T>
void AdadeltaOptimizer<T>::set_weight(const Tensor<T> *p) {
  size_t size = p->width();
  T* gptr = new T[size];
  accum_gradient = Tensor<T>(gptr, size);
  T* dptr = new T[size];
  accum_delta = Tensor<T>(dtpr, size);
  T* dptr_current = new T[size];
  update_delta = Tensor<T>(dptr_current, size);
}

template<class T>
void AdadeltaOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate();
  for(size_t i=0; i<parameter_.size(); ++i) {
    accum_gradient[i] = rho * accum_gradient[i] + (1.0 - rho) * gradient[i] * gradient[i];
    
    update_delta[i] = std::sqrt(accum_delta[i] + epsilon) / std::sqrt(accum_gradient[i] + epsilon) * gradient[i];

    accum_delta[i] = rho * accum_delta[i] + (1.0-rho) * update_delta[i] * update_delta[i];

    parameter_[i] -= update_delta[i] + decay*parameter_[i];
  }
}


template class AdadeltaOptimizer<float>;
template class AdadeltaOptimizer<double>;

}
}
