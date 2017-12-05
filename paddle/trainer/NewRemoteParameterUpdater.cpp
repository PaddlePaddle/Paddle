/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "NewRemoteParameterUpdater.h"
#include "Trainer.h"
#include "paddle/utils/Stat.h"

DECLARE_int32(trainer_id);
DECLARE_string(save_dir);

namespace paddle {
NewRemoteParameterUpdater::NewRemoteParameterUpdater(
    const OptimizationConfig &config, const std::string pserverSpec)
    : trainerConfig_(config),
      parameterClient_(-1),
      newParameters_(nullptr),
      newGradients_(nullptr),
      pserverSpec_(pserverSpec) {}

NewRemoteParameterUpdater::NewRemoteParameterUpdater(
    const OptimizationConfig &config,
    const std::string pserverSpec,
    const bool useEtcd)
    : trainerConfig_(config),
      parameterClient_(-1),
      newParameters_(nullptr),
      newGradients_(nullptr),
      pserverSpec_(pserverSpec),
      useEtcd_(useEtcd) {}

void NewRemoteParameterUpdater::init(
    const std::vector<ParameterPtr> &parameters) {
  ParameterUpdater::init(parameters);

  // create parameter server client.
  if (useEtcd_) {
    parameterClient_ =
        paddle_new_etcd_pserver_client((char *)pserverSpec_.c_str());
  } else {
    parameterClient_ = paddle_new_pserver_client((char *)pserverSpec_.c_str(),
                                                 FLAGS_trainer_id == 0);
  }

  // init new parameter and gradient.
  newParameters_ = initNewParameter(PARAMETER_VALUE);
  newGradients_ = initNewParameter(PARAMETER_GRADIENT);

  // init parameter, one trainer will get the opportunity to int parameter and
  // send them to parameter server. Others will get the initialized parameter
  // from parameter server
  if (paddle_begin_init_params(parameterClient_)) {
    LOG(INFO) << "paddle_begin_init_params start";
    // NOTE: convert V1 OptimizatioinConfig proto to V2 OptimizerConfig.
    // This makes golang pserver compatible with handy V1 demos.
    // TODO(wuyi): Refine or remove these ugly converting lines
    OptimizerConfig optimizerConfigV2;
    if (trainerConfig_.learning_method() == "momentum") {
      optimizerConfigV2.set_optimizer(paddle::OptimizerConfig::SGD);
    } else if (trainerConfig_.learning_method() == "adagrad") {
      optimizerConfigV2.set_optimizer(paddle::OptimizerConfig::Adagrad);
      optimizerConfigV2.mutable_adagrad()->set_epsilon(
          trainerConfig_.ada_epsilon());
    } else if (trainerConfig_.learning_method() == "adadelta") {
      optimizerConfigV2.set_optimizer(paddle::OptimizerConfig::Adagrad);
      optimizerConfigV2.mutable_adadelta()->set_epsilon(
          trainerConfig_.ada_epsilon());
      optimizerConfigV2.mutable_adadelta()->set_rho(trainerConfig_.ada_rou());
    } else if (trainerConfig_.learning_method() == "adam") {
      optimizerConfigV2.set_optimizer(paddle::OptimizerConfig::Adam);
      optimizerConfigV2.mutable_adam()->set_beta_1(trainerConfig_.adam_beta1());
      optimizerConfigV2.mutable_adam()->set_beta_2(trainerConfig_.adam_beta2());
      optimizerConfigV2.mutable_adam()->set_epsilon(
          trainerConfig_.adam_epsilon());
    } else {
      LOG(ERROR) << "got unsupported v1 optimizer config: "
                 << trainerConfig_.learning_method();
      optimizerConfigV2.set_optimizer(paddle::OptimizerConfig::SGD);
    }

    if (trainerConfig_.learning_rate_schedule() == "constant") {
      optimizerConfigV2.set_lr_policy(paddle::OptimizerConfig::Const);
      optimizerConfigV2.mutable_const_lr()->set_learning_rate(
          trainerConfig_.learning_rate());
    } else if (trainerConfig_.learning_rate_schedule() == "linear") {
      optimizerConfigV2.set_lr_policy(paddle::OptimizerConfig::Linear);
      optimizerConfigV2.mutable_linear_lr()->set_learning_rate(
          trainerConfig_.learning_rate());
      optimizerConfigV2.mutable_linear_lr()->set_lr_decay_a(
          trainerConfig_.learning_rate_decay_a());
      optimizerConfigV2.mutable_linear_lr()->set_lr_decay_b(
          trainerConfig_.learning_rate_decay_b());
    } else {
      LOG(ERROR) << "got unsupported v1 learning_rate_schedule config: "
                 << trainerConfig_.learning_rate_schedule() << ", set to const";
      optimizerConfigV2.set_lr_policy(paddle::OptimizerConfig::Const);
      optimizerConfigV2.mutable_const_lr()->set_learning_rate(
          trainerConfig_.learning_rate());
    }

    // overwrite optimizerConfigV2 for per-parameter(layer) configs
    for (int i = 0; i < parameterSize(); ++i) {
      // FIXME(typhoonzero): paramConfig always have default values,
      // how to check if it's default?
      // TODO(typhoonzero): log output: optimizerConfigV2.DebugString();
      LOG(INFO) << "trainerConfig_: " << trainerConfig_.DebugString();
      // send param and config to pserver
      std::string bytes = optimizerConfigV2.SerializeAsString();
      const char *array = bytes.data();
      int size = (int)bytes.size();
      paddle_init_param(
          parameterClient_, *newParameters_[i], (void *)array, size);
    }
    paddle_finish_init_params(parameterClient_);
    LOG(INFO) << "paddle_begin_init_params done";
  } else {
    paddle_get_params(parameterClient_, newParameters_, parameterSize());
  }

  LOG(INFO) << "NewRemoteParameterUpdater initialized";
}

void NewRemoteParameterUpdater::updateImpl(Parameter *para) {}

void NewRemoteParameterUpdater::finishBatch(real cost) {
  // send gradient to parameter server.
  paddle_send_grads(parameterClient_, newGradients_, parameterSize());
  // get the updated parameter from parameterClient.
  paddle_get_params(parameterClient_, newParameters_, parameterSize());

  // clear gradient after update parameter.
  for (auto &para : parameters_) {
    para->getBuf(PARAMETER_GRADIENT)->zeroMem();
  }
}

void NewRemoteParameterUpdater::startPass() {}

bool NewRemoteParameterUpdater::finishPass() { return true; }
}  // namespace paddle
