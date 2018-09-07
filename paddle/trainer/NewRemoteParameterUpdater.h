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

#pragma once

#include <functional>
#include <thread>
#include "OptimizerConfig.pb.h"
#include "ParameterUpdater.h"
#include "libpaddle_pserver_cclient.h"
#include "paddle/pserver/ParameterClient2.h"
#include "paddle/utils/Queue.h"
#include "paddle/utils/Util.h"

namespace paddle {

/**
 * New remote parameter updater for dense parameters that use cclient of go.
 */
class NewRemoteParameterUpdater : public ParameterUpdater {
 public:
  NewRemoteParameterUpdater(const OptimizationConfig& config,
                            const std::string pserverSpec);
  NewRemoteParameterUpdater(const OptimizationConfig& config,
                            const std::string pserverSpec,
                            const bool useEtcd);
  ~NewRemoteParameterUpdater() {
    releaseNewParameter(newParameters_);
    releaseNewParameter(newGradients_);
    if (parameterClient_ >= 0) paddle_pserver_client_release(parameterClient_);
  }

  /**
   * initialize the internal parameter client and itself.
   */
  virtual void init(const std::vector<ParameterPtr>& parameters);
  /**
   * @brief start batch
   *
   * @note  one batch training exhibits stateful feature to help
   *        to do performance tuning, sgd optimization if necessary.
   */
  virtual PassType startBatch(int64_t batchSize) { return PASS_TRAIN; }

  /**
   * send parameters to pservers and get returned parameters
   * from all pservers if necessary.
   */
  virtual void finishBatch(real cost);
  virtual void startPass();
  virtual bool finishPass();

 protected:
  /**
   * work need to do after finishBatch
   */
  virtual void updateImpl(Parameter* para);

 private:
  int parameterSize() { return (int)parameters_.size(); }

  /**
   * init parameter of go paddle pserver cclient.
   * @param new_params
   * @param type
   */
  paddle_parameter** initNewParameter(ParameterType type) {
    paddle_parameter** new_params =
        (paddle_parameter**)malloc(sizeof(paddle_parameter*) * parameterSize());
    for (int i = 0; i < parameterSize(); ++i) {
      new_params[i] = (paddle_parameter*)malloc(sizeof(paddle_parameter));
      memset(new_params[i], 0, sizeof(paddle_parameter));
    }

    for (int i = 0; i < parameterSize(); ++i) {
      ParameterPtr param = parameters_[i];
      new_params[i]->element_type = PADDLE_ELEMENT_TYPE_FLOAT32;
      new_params[i]->name = (char*)param->getName().c_str();
      new_params[i]->content =
          (unsigned char*)(param->getBuf(type).get()->getData());
      new_params[i]->content_len =
          (int)param->getBuf(type).get()->getSize() * sizeof(real);
    }
    return new_params;
  }

  void releaseNewParameter(paddle_parameter** newParams) {
    if (newParams != nullptr) {
      for (int i = 0; i < parameterSize(); ++i) {
        free(newParams[i]);
      }
      free(newParams);
    }
  }

 protected:
  const OptimizationConfig& trainerConfig_;
  /// internal parameter client object for exchanging data with pserver
  paddle_pserver_client parameterClient_;
  /// the parameters for new pserver client
  paddle_parameter** newParameters_;
  /// the gradinets for new pserver client
  paddle_parameter** newGradients_;
  /// the specification of parameter server "host1:port,host1:port"
  std::string pserverSpec_;
  /// true if pserverSpec_ is etcd endpoint, else pserverSpec_ is pserver addr
  bool useEtcd_;
};

}  // namespace paddle
