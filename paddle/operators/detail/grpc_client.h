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

#include <time.h>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "paddle/framework/data_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/operators/detail/simple_block_queue.h"

#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>

using grpc::Channel;

namespace paddle {
namespace operators {
namespace detail {

struct Var {
  std::string endpoint;
  std::string name;
};

struct SendStatus {
  std::string error;
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  Var var;
};

class AsyncGRPCClient {
 public:
  AsyncGRPCClient() {}

  void AddEndPoint(std::string ep);

  bool SendVariable(const framework::Scope& scope, std::vector<Var>& in,
                    std::vector<SendStatus>& ret);

  bool GetVariable(const framework::Scope& scope, std::vector<Var>& in,
                   std::vector<SendStatus>& ret);

  // TODO(gongwb): add SendRecv function to try to update
  // one local parameter immediately when it's gradient
  // is sent completely and don't wait all.
 protected:
  template <typename send_t, typename recv_t, typename Msg_t>
  bool call(const framework::Scope& scope, std::vector<Var>& in,
            std::vector<SendStatus>& ret);

 private:
  std::map<std::string, std::shared_ptr<Channel>> channels_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
