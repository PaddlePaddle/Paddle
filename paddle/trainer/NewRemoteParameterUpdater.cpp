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

#include <go/pserver/cclient/libclient.h>
#include "NewRemoteParameterUpdater.h"
#include "Trainer.h"
#include "paddle/utils/Stat.h"

DECLARE_int32(trainer_id);
DECLARE_string(save_dir);

namespace paddle {
    NewRemoteParameterUpdater::NewRemoteParameterUpdater(
            const OptimizationConfig &config) {}

    void NewRemoteParameterUpdater::init(const std::vector<ParameterPtr> &parameters) {
      ParameterUpdater::init(parameters);

      // create parameter server client.
      char addr[] = "localhost:3000";
      parameterClient_ = paddle_new_pserver_client(addr, FLAGS_trainer_id);

      // init parameter of new cclient.
      *newParameters_ = new paddle_parameter(parameters.size());
      for (int i = 0; i < parameters.size(); ++i) {
        auto& para = parameters[i];
        newParameters_[i]->element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
        newParameters_[i]->name = para->getName().c_str();
        newParameters_[i]->content =
                (unsigned char *)(para->getBuf(PARAMETER_VALUE).get()->getData());
        newParameters_[i]->content_len = (int)para->getBuf(PARAMETER_VALUE).get()->getSize();
      }

      *names_ = new const char(parameters_.size());
      for (int i = 0; i < parameters_.size(); ++i) {
        names_[i] = parameters_[i]->getName().c_str();
      }

      // init parameter, one trainer will get the opportunity to int parameter and send
      // them to parameter server. Others will get the initialized parameter from parameter
      // server
      if (paddle_begin_init_params(parameterClient_)) {
        for (int i = 0; i < parameters_.size(); ++i) {
          paddle_init_param(parameterClient_, *newParameters_[i], NULL, 0);
        }
      } else {
        paddle_get_params(parameterClient_, names_, newParameters_, (int)parameters_.size());
      }
      paddle_finish_init_params(parameterClient_);
    }

    void NewRemoteParameterUpdater::updateImpl(Parameter *para) {
    }

    void copyToParameters() {
    }

    void NewRemoteParameterUpdater::finishBatch(real cost) {

      // send parameter to parameter server.
      for (int i = 0; i < (int)parameters_.size(); ++i) {
        auto para = newParameters_[i];
        paddle_send_grads(parameterClient_, para, para->content_len);
      }

      // get the updated parameter from parameterClient.
      paddle_get_params(parameterClient_, names_, newParameters_, (int)parameters_.size());

      // clear gradient after update parameter.
      for (auto& para : parameters_) {
        SetDevice device(para->getDeviceId());
        para->getBuf(PARAMETER_GRADIENT)->zeroMem();
      }
    }

    void NewRemoteParameterUpdater::startPass() {
    }

    bool NewRemoteParameterUpdater::finishPass() {
      return true;
    }
}