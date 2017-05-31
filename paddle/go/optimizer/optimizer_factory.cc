#include "optimizer_factory.h"

namespace paddle {
namespace optimizer {

template<class T>
SGDOptimizer<T>::SGDOptimizer(const OptimizerConfig &config) : ParameterOptimizer<T>(config) {
  learning_rate = config.learning_rate;
  momentum = config.momentum;
  decay = config.decay;
  nesterov = config.nesterov;
}

template<class T>
void SGDOptimizer<T>::destroy() {
  ~SGDOptimizer();
}

template<class T>
void SGDOptimizer<T>::set_weight(const Tensor<T> *p) {
  ParameterOptimizer::set_weight(p);
  size_t size = p->height();
  // TODO: fix it with align aware allocator bind to Tensor
  T* ptr = new T[size];
  momentum_ = Tensor<T>(ptr, size);
  
}

template<class T>
void SGDOptimizer<T>::update(const Tensor<T> &gradient) {
  learning_rate = applyLinearLearningRate(config_);
  for(size_t i=0; i<parameter_.size(); ++i) {
    momentums_[i] = momentum * momentums_[i] - learning_rate*gradient[i] - decay*parameter_[i];
    if(nesterov) {
      //TODO(zhihong) : fix nesterov
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
template class AdagradOptimizer<float>;
template class AdagradOptimizer<double>;
template class AdadeltaOptimizer<float>;
template class AdadeltaOptimizer<double>;
template class AdamOptimizer<float>;
template class AdamOptimizer<double>;



// template<class T>
// AdagradOptimizer(OptimizerConfig &config) : ParameterOptimizer(config) {
//   learning_rate = config.learning_rate;
//   momentum = config.momentum;
//   decay = config.decay;
//   nesterov = confg.nesterov;
// }




// template <class T>
// MomentumOptimizer<T>::MomentumOptimizer(const paddle::optimizer_config &config)
//     : ParameterOptimizer(config) {
//   momentum = config.mometum;
// }

}  // namespace optimizer
}  // namespace paddle
