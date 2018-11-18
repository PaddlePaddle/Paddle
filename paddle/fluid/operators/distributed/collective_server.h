/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {
class CollectiveSever final {
 public:
  explicit CollectiveSever(const std::string& end_point, int fan_in);

  virtual ~CollectiveSever() {}

  void StartServer();

  static CollectiveSever* GetInstance(const std::string& end_point,
                                      int fan_in) {
    std::call_once(init_flag_, [&]() {
      if (collective_server_.get() == nullptr) {
        collective_server_.reset(new CollectiveSever(end_point, fan_in));
      }
    });

    return collective_server_.get();
  }

 private:
  std::shared_ptr<distributed::RPCServer> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;

  static std::once_flag init_flag_;
  static std::unique_ptr<CollectiveSever> collective_server_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
