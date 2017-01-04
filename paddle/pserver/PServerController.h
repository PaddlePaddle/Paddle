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

class PServerController final {
public:
  DISABLE_COPY(PServerController);

  /**
   * @brief Ctor, Create a PServerController from ParameterServerConfig.
   */
  explicit PServerController(const ParameterServerConfig& config);

  /**
   * @brief Dtor.
   */
  ~PServerController();

  /**
   * @brief create PServerController from gflags, this is used for
   * compatibility with the old usage of configuration by gflags.
   */
  static PServerController* createByGflags();

  /**
   * @brief create PServerController with ParameterServerConfig, remove gflags
   * from ParameterServer. Init all pservers thread according to the config.
   */
  static PServerController* create(const ParameterServerConfig& config);

  /**
   * @brief start all pserver thread in this PServerController.
   */
  void start();

  /**
   * @brief join and wait for all pserver thread in this PServerController.
   */
  void join();

private:
  std::vector<std::unique_ptr<ParameterServer2>> pservers_;
};

}  // namespace paddle
