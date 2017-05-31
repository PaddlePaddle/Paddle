#include "parameter_optimizer.h"
#include "optimizer_factory.h"
#include <glog/logging.h>

namespace paddle {
namespace optimizer {

template <class T>
ParameterOptimizer<T>* ParameterOptimizer<T>::create(
    const std::string &config_proto) {
  paddle::OptimizerConfig config;
  CHECK(config.ParseFromString(config_proto) == 0) << "error : optimizer config";
  CHECK(config_valid(config) == 0) << "error : invalid optimizer config ";
  ParameterOptimizer<T> *opt = nullptr;
  switch (config.optimizer_name) {
  case "SGD" : opt = new SGDOptimizer<T>(config); break;
  case "Adagrad" : opt = new AdagradOptimizer<T>(config); break;
  case "Adadelta" : opt = new AdadeltaOptimizer<T>(config); break;
  case "Adam" : opt = new AdamOptimizer<T>(config); break;
  default:
    opt = new SGDOptimizer<T>(config);
  }
  return opt;
}

template<class T>
double ParameterOptimzier<T>::get_learning_rate() {
  
}

template <class T>
T* ParameterOptimizer<T>::get_weight() const {
  return parameter.get().get_buffer();
}

template <class T>
char* ParameterOptimizer<T>::get_config_proto() const {
  // set config dynamic value for save checkpoint
  config_.set_num_sample_passed(num_sample_passed);
  config_.set_iterations(iterations);
  return config_.SerializeAsString().c_str();
}

template <class T>
void ParameterOptimizer<T>::set_weight(const Tensor<T> *p) {
  parameter_ = p;
}

template<class T>
bool ParameterOptimizer<T>::config_valid(const std::string &config) const{
  
  // TODO(zhihong) : add more value checker, failed ASAP
  return true;
}

template class ParameterOptimzier<float>;
template class ParameterOptimzier<double>;

}  // namespace optimizer
}  // namespace paddle
