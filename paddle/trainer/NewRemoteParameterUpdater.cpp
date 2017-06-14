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
    : parameterClient_(-1),
      newParameters_(nullptr),
      newGradients_(nullptr),
      pserverSpec_(pserverSpec) {}

void NewRemoteParameterUpdater::init(
    const std::vector<ParameterPtr> &parameters) {
  ParameterUpdater::init(parameters);

  for (auto &para : parameters_) {
    para->getBuf(PARAMETER_VALUE)->zeroMem();
    para->getBuf(PARAMETER_GRADIENT)->zeroMem();
  }

  // create parameter server client.
  parameterClient_ = paddle_new_pserver_client((char *)pserverSpec_.c_str(),
                                               FLAGS_trainer_id == 0);

  // init new parameter and gradient.
  newParameters_ = initNewParameter(PARAMETER_VALUE);
  newGradients_ = initNewParameter(PARAMETER_GRADIENT);

  // init parameter, one trainer will get the opportunity to int parameter and
  // send them to parameter server. Others will get the initialized parameter
  // from parameter server
  if (paddle_begin_init_params(parameterClient_)) {
    LOG(INFO) << "paddle_begin_init_params start";
    for (int i = 0; i < parameterSize(); ++i) {
      auto paramConfig = parameters_[i]->getConfig();
      std::string bytes = paramConfig.SerializeAsString();
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
}
