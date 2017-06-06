#include <glog/logging.h>
#include "adadelta_optimizer.h"
#include "adagrad_optimizer.h"
#include "adam_optimizer.h"
#include "lr_policy.h"
#include "sgd_optimizer.h"

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

ParameterOptimizer *ParameterOptimizer::Create(
    const std::string &config_proto) {
  paddle::OptimizerConfig config;
  CHECK(config.ParseFromString(config_proto) == 0)
      << "failed parse optimizer config";

  auto select_lr_policy = [=](const OptimizerConfig &config) -> LrPolicy * {
    if (config.lr_policy() == OptimizerConfig::ConstLr)
      return new ConstLr(config.const_lr().learning_rate());
    if (config.lr_policy() == OptimizerConfig::LinearLr)
      return new LinearLr(config.linear_lr().learning_rate(),
                          config.linear_lr().lr_decay_a(),
                          config.linear_lr().lr_decay_b());
    // default
    return nullptr;
  };
  LrPolicy *lr = select_lr_policy(config);
  auto select_optimizer =
      [=](const OptimizerConfig &config) -> ParameterOptimizer * {
    if (config.optimizer() == OptimizerConfig::SGD) {
      return new SGDOptimizer(config.sgd().momentum(),
                              config.sgd().decay(),
                              config.sgd().nesterov(),
                              lr);
    }
    if (config.optimizer() == OptimizerConfig::Adadelta) {
      return new AdagradOptimizer(
          config.adagrad().epsilon(), config.adagrad().decay(), lr);
    }
    if (config.optimizer() == OptimizerConfig::Adagrad) {
      return new AdagradOptimizer(
          config.adagrad().epsilon(), config.adagrad().decay(), lr);
    }
    if (config.optimizer() == OptimizerConfig::Adam) {
      return new AdadeltaOptimizer(config.adadelta().rho(),
                                   config.adadelta().epsilon(),
                                   config.adadelta().decay(),
                                   lr);
    }
    // default
    return new SGDOptimizer(config.sgd().momentum(),
                            config.sgd().decay(),
                            config.sgd().nesterov(),
                            lr);
  };
  return select_optimizer(config);
}

real *ParameterOptimizer::get_weight() const {
  return parameter_->get_buffer();
}

void ParameterOptimizer::set_weight(Tensor *p) { parameter_ = p; }

}  // namespace optimizer
}  // namespace paddle
