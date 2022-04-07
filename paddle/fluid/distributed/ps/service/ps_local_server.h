// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>
#include "paddle/fluid/distributed/ps/service/server.h"

namespace paddle {
namespace distributed {

class PsLocalServer : public PSServer {
 public:
  PsLocalServer() {}
  virtual ~PsLocalServer() {}
  virtual uint64_t Start() { return 0; }
  virtual uint64_t Start(const std::string &ip, uint32_t port) { return 0; }
  virtual int32_t Stop() { return 0; }
  virtual int32_t Configure(
      const PSParameter &config, PSEnvironment &env, size_t server_rank,
      const std::vector<framework::ProgramDesc> &server_sub_program = {}) {
    return 0;
  }

 private:
  virtual int32_t Initialize() { return 0; }
};
}
}
