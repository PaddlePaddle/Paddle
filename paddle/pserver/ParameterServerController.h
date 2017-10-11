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

#include "ParameterServer2.h"
#include "ParameterServerConfig.pb.h"
#include "RDMANetwork.h"
#include "paddle/utils/StringUtil.h"

namespace paddle {

/**
 * @brief ParameterServerController is used for create, init and manage multi
 * parameter server instances. The num of the instances is decided by port
 * num(the ports number for parameter send) and network devices configured
 * by gflags or proto.
 */
class ParameterServerController final {
public:
  DISABLE_COPY(ParameterServerController);

  /**
   * @brief Ctor, Create a ParameterServerController from ParameterServerConfig.
   */
  explicit ParameterServerController(const ParameterServerConfig& config);

  /**
   * @brief Dtor.
   */
  ~ParameterServerController();

  /**
   * @brief create ParameterServerController from gflags, this is used for
   * compatibility with the old usage of configuration by gflags.
   */
  static ParameterServerController* createFromGflags();

  /**
   * @brief create ParameterServerController with ParameterServerConfig, remove
   * gflags from ParameterServer. Init all ParameterServer2 instances according
   * to
   * the config.
   */
  static ParameterServerController* create(const ParameterServerConfig& config);

  /**
   * @brief start all ParameterServer2 instances in this
   * ParameterServerController.
   */
  void start();

  /**
   * @brief join and wait for all ParameterServer2 instances thread in this
   * ParameterServerController.
   */
  void wait();

private:
  std::vector<std::unique_ptr<ParameterServer2>> parameterServers_;
};

}  // namespace paddle
