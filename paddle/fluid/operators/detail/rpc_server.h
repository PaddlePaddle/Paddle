// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>
#include "paddle/fluid/operators/detail/request_handler.h"

namespace paddle {
namespace operators {
namespace detail {

class RPCServer {
 public:
  explicit RPCServer(const std::string& address,
                     RequestHandler* request_handler)
      : address_(address),
        request_handler_(request_handler),
        selected_port_(-1) {}

  virtual ~RPCServer() {}
  virtual void WaitServerReady() = 0;
  virtual void RunSyncUpdate() = 0;
  virtual int GetSelectedPort() const { return selected_port_; }
  virtual void ShutDown() = 0;
  virtual void RegisterCond(int rpc_id) = 0;
  virtual void SetCond(int rpc_id) = 0;

 protected:
  virtual void WaitCond(int cond) = 0;

 protected:
  std::string address_;
  RequestHandler* request_handler_;
  std::set<int> cond_;
  int selected_port_;
};

};  // namespace detail
};  // namespace operators
};  // namespace paddle
