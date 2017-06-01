#include "sgd_optimizer.h"

namespace paddle {
namespace optimizer {

template<class T>
SGDOptimizer<T>::SGDOptimizer(const ::paddle::OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  learning_rate = config.learning_rate();
  momentum = config.momentum();
  decay = config.decay();
  nesterov = config.nesterov();
  lr_decay_a = config.lr_decay_a();
  lr_decay_b = config.lr_decay_b();
}

template<class T>
void SGDOptimizer<T>::set_weight(const Tensor<T> *p) {
//  ParameterOptimizer::set_weight(p);
  size_t size = p->height();
  // TODO: fix it with align aware allocator bind to Tensor
  T* ptr = new T[size];
  momentum_ = Tensor<T>(ptr, size);
  
}

template<class T>
void SGDOptimizer<T>::update(const Tensor<T> &gradient) {
  num_sample_passed += 1;
  learning_rate = get_learning_rate();
  for(size_t i=0; i<parameter_.size(); ++i) {
    momentums_[i] = momentum * momentums_[i] - learning_rate*gradient[i] - decay*parameter_[i];
    if(nesterov) {
      //TODO(zhihong) : fix nesterov updating
      parameter_[i] += momentums_[i];
    } else {
      parameter_[i] += momentums_[i];
    }
  }
}

template<class T>
char* SGDOptimizer<T>::get_config_proto() {
  ParameterOptimizer::get_config_proto();
  config.set_learning_rate(learning_rate);
  config.set_momentum(momentum);
  config.set_decay(decay);
  config.set_nesterov(nesterov);
  return config.SerializeAsString().c_str();
}

template class SGDOptimizer<float>;
template class SGDOptimizer<double>;

}  // namespace optimizer
}  // namespace paddle
