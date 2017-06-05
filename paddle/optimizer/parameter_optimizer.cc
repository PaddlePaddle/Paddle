#include <glog/logging.h>
#include "adadelta_optimizer.h"
#include "adagrad_optimizer.h"
#include "adam_optimizer.h"
#include "lr_policy.h"
#include "sgd_optimizer.h"

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

template <class T>
ParameterOptimizer<T> *ParameterOptimizer<T>::create(
    const ::std::string &config_proto) {
  paddle::OptimizerConfig config;
  CHECK(config.ParseFromString(config_proto) == 0)
      << "error : optimizer config";
  CHECK(config_valid(config) == 0) << "error : invalid optimizer config ";

  BaseLr *lr = nullptr;
  switch (config.lr_policy()) {
    case "ConstLr":
      lr = new ConstLr(config.lr_config().learning_rate());
      break;
  }
  ParameterOptimizer<T> *opt = nullptr;
  switch (config.optimizer_name()) {
    case "SGD":
      opt = new SGDOptimizer<T>(config.sgd().momentum(),
                                config.sgd().decay(),
                                config.sgd().nesterov(),
                                lr);
      break;
    case "Adagrad":
      opt = new AdagradOptimizer<T>(
          config.adagrad().epsilon(), config.adagrad().decay(), lr);
      break;
    case "Adadelta":
      opt = new AdadeltaOptimizer<T>(config.adadelta().rho(),
                                     config.adadelta().epsilon(),
                                     config.adadelta().decay(),
                                     lr);
      break;
    case "Adam":
      opt = new AdamOptimizer<T>(config.adam().beta_1(),
                                 config.adam().beta_2(),
                                 config.adam().epsilon(),
                                 config.adam().decay(),
                                 lr);
      break;
  }

  return opt;
}

template <class T>
T *ParameterOptimizer<T>::get_weight() const {
  return parameter.get().get_buffer();
}

template <class T>
char *ParameterOptimizer<T>::get_config_proto() const {
  // set config dynamic value for save checkpoint
  config_.lr_policy().set_learning_rate(
      lr_policy->get_learning_rate(num_sample_passed));
  config_.set_num_sample_passed(num_sample_passed);
  config_.set_iterations(iterations);
  return config_.SerializeAsString().c_str();
}

template <class T>
void ParameterOptimizer<T>::set_weight(const Tensor<T> *p) {
  parameter_ = p;
}

template <class T>
bool ParameterOptimizer<T>::config_valid(const ::std::string &config) const {
  // TODO(zhihong) : add more value checker, failed ASAP
  return true;
}

template class ParameterOptimzier<float>;
template class ParameterOptimzier<double>;

}  // namespace optimizer
}  // namespace paddle
