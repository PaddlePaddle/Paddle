#include "sgd_optimizer.h"

namespace paddle {
namespace optimizer {

template<class T>
SGDOptimizer<T>::SGDOptimizer(const ::paddle::OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  momentum = config.sgd().momentum();
  decay = config.sgd().decay();
  nesterov = config.sgd().nesterov();
}

template<class T>
void SGDOptimizer<T>::set_weight(const Tensor<T> *p) {
//  ParameterOptimizer::set_weight(p);
  size_t size = p->width();
  // TODO: fix it with align aware allocator bind to Tensor
  if(momentum != 0.0) {
    T* ptr = new T[size];
    momentums_ = Tensor<T>(ptr, size);
  }
}

template<class T>
void SGDOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  double learning_rate = lr_policy->get_learning_rate(num_sample_passed);
  double velocity = 0.0;
  for(size_t i=0; i<parameter_.size(); ++i) {
    if(momentum == 0.0) {
      velocity = -learning_rate*gradient[i] - learning_rate*decay*parameter_[i];
    } else {
      momentums_[i] = momentum * momentums_[i] - learning_rate*gradient[i]
        - learning_rate*decay*parameter_[i];
      velocity = momentums_[i];
    }
    if(nesterov) {
      parameter_[i] += momentum*velocity - learning_rate*gradient[i];
    } else {
      parameter_[i] += velocity;
    }
  }
}

template<class T>
char* SGDOptimizer<T>::get_config_proto() {
  ParameterOptimizer::get_config_proto();
  config.set_learning_rate(learning_rate);
  config.set_decay(decay);
  config.set_nesterov(nesterov);
  return config.SerializeAsString().c_str();
}

template class SGDOptimizer<float>;
template class SGDOptimizer<double>;

}  // namespace optimizer
}  // namespace paddle
