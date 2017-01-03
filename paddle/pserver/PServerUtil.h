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

class PServerUtil {
public:
  DISABLE_COPY(PServerUtil);
  static PServerUtil* create();
  static PServerUtil* create(const ParameterServerConfig& config);
  explicit PServerUtil(const ParameterServerConfig& config);
  ~PServerUtil();
  static ParameterServerConfig* initConfig();
  void start();
  void join();

private:
  std::vector<std::shared_ptr<ParameterServer2>> pservers_;
};

}  // namespace paddle
