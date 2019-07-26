/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_service.h"
#include "paddle/fluid/operators/distributed/request.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

class GRPCRequest;

class AsyncGRPCServer final : public RPCServer {
 public:
  explicit AsyncGRPCServer(const std::string& address, int client_num)
      : RPCServer(address, client_num), ready_(0) {}

  virtual ~AsyncGRPCServer() {}
  void WaitServerReady() override;
  void StartServer() override;

 private:
  // HandleRequest needs to be thread-safe.
  void HandleRequest(::grpc::ServerCompletionQueue* cq, RequestType rpc_type,
                     std::function<void(RequestType, int)> TryToRegisterNewOne);

  void TryToRegisterNewOne(const RequestType rpc_name, int req_id);
  void ShutdownQueue();
  void ShutDownImpl() override;

 private:
  static const int kRequestBufSize = 100;

  std::mutex cq_mutex_;
  volatile bool is_shut_down_ = false;

  GrpcService::AsyncService service_;
  std::unique_ptr<::grpc::Server> server_;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;

  int ready_;

  std::unordered_map<RequestType,
                     std::unique_ptr<::grpc::ServerCompletionQueue>,
                     EnumClassHash>
      rpc_cq_;
  std::unordered_map<RequestType, std::vector<std::unique_ptr<std::thread>>,
                     EnumClassHash>
      rpc_threads_;
  std::unordered_map<RequestType, std::vector<GRPCRequest*>, EnumClassHash>
      rpc_reqs_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
