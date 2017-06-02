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
#include "ParameterUpdater.h"
#include "paddle/pserver/ParameterClient2.h"
#include "paddle/utils/Queue.h"
#include "paddle/utils/Util.h"
#include "go/pserver/cclient/libclient.h"

namespace paddle {

/**
 * New remote parameter updater for dense parameters that use cclient of go.
 */
class NewRemoteParameterUpdater : public ParameterUpdater {
public:
  NewRemoteParameterUpdater(const OptimizationConfig& config);
  ~NewRemoteParameterUpdater() {}

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
  virtual PassType startBatch(int64_t batchSize) {
    return PASS_TRAIN;
  }

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

protected:
  /// internal parameter client object for exchanging data with pserver
  client parameterClient_;
  /// the parameters for new pserver client
  paddle_parameter** newParameters_;
  /// the names for new parameters.
  const char** names_;

  static const std::string kAverage;
  static const std::string kElasticAverage;
};

}  // namespace paddle
