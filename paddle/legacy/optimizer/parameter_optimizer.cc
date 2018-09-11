/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include "adadelta_optimizer.h"
#include "adagrad_optimizer.h"
#include "adam_optimizer.h"
#include "lr_policy.h"
#include "sgd_optimizer.h"

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

ParameterOptimizer *ParameterOptimizer::Create(const std::string &config_proto,
                                               Tensor *parameter) {
  paddle::OptimizerConfig config;
  CHECK(config.ParseFromString(config_proto) == true)
      << "failed parse optimizer config";
  auto select_lr_policy = [=](const OptimizerConfig &config) -> LrPolicy * {
    if (config.lr_policy() == OptimizerConfig::Const)
      return new ConstLr(config.const_lr().learning_rate());
    if (config.lr_policy() == OptimizerConfig::Linear)
      return new LinearLr(config.linear_lr().learning_rate(),
                          config.linear_lr().lr_decay_a(),
                          config.linear_lr().lr_decay_b());
    // default
    LOG(WARNING) << " have not select any LrPolicy. use ConstLr in default";
    return new ConstLr(0.1);
  };

  LrPolicy *lr = select_lr_policy(config);
  auto select_optimizer = [=](
      Tensor *parameter,
      const OptimizerConfig &config) -> ParameterOptimizer * {
    if (config.optimizer() == OptimizerConfig::SGD) {
      LOG(INFO) << "creating SGD optimizer";
      return new SGDOptimizer(parameter,
                              lr,
                              config.sgd().momentum(),
                              config.sgd().decay(),
                              config.sgd().nesterov());
    }
    if (config.optimizer() == OptimizerConfig::Adadelta) {
      LOG(INFO) << "creating Adadelta optimizer";
      return new AdadeltaOptimizer(parameter,
                                   lr,
                                   config.adadelta().rho(),
                                   config.adadelta().epsilon(),
                                   config.adadelta().decay());
    }
    if (config.optimizer() == OptimizerConfig::Adagrad) {
      LOG(INFO) << "creating Adagrad optimizer";
      return new AdagradOptimizer(
          parameter, lr, config.adagrad().epsilon(), config.adagrad().decay());
    }
    if (config.optimizer() == OptimizerConfig::Adam) {
      LOG(INFO) << "creating Adam optimizer";
      return new AdamOptimizer(parameter,
                               lr,
                               config.adam().beta_1(),
                               config.adam().beta_2(),
                               config.adam().epsilon(),
                               config.adam().decay());
    }
    // default
    LOG(WARNING)
        << "have not select any Optimizer. use SGDOptimizer in default";
    return new SGDOptimizer(parameter, lr, 0.0, 0.0, false);
  };
  return select_optimizer(parameter, config);
}

float *ParameterOptimizer::get_weight(int *param_size) const {
  *param_size = (int)parameter_->size();
  return parameter_->get_buffer();
}

}  // namespace optimizer
}  // namespace paddle
