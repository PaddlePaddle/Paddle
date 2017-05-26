#include "parameter_optimizer.h"
#include "momentum_optimizer.h"
#include "sgd_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
static ParameterOptimizer<T> *ParameterOptimizer<T>::create(
    const std::string &config_proto, Tensor<T> *parameter) {
  paddle::optimizer_config config;
  CHECK(config.ParseFromString(config_proto) == 0)
      << "parse optimizer config error";
  ParameterOptimizer *opt;
  if (config.optimizer_name == "SGDOptimizer") {
    opt = new SGDOptimizer(config);
  } else if (config.optimizer_name == "MomentumOptimizer") {
    opt = new MomentumOptimizer(config);
  }
  opt.set_buffer(parameter)
}

template <class T>
T *ParameterOptimizer<T>::get_buffer() const {
  return parameter.get().get_buffer();
}

template <class T>
std::string ParameterOptimizer<T>::get_config_proto() const {
  return config_.SerializeAsString();
}

template <class T>
void ParameterOptimizer<T>::set_buffer(const Tensor<T> *p) {
  parameter.reset(p);
}

}  // namespace optimizer
}  // namespace paddle
