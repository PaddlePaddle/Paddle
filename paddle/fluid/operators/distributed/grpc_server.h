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

#include "grpc++/grpc++.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/grpc_service.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/operators/distributed/send_recv.grpc.pb.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

class RequestBase;

class AsyncGRPCServer final : public RPCServer {
 public:
  explicit AsyncGRPCServer(const std::string& address, int client_num)
      : RPCServer(address, client_num), ready_(0) {}

  virtual ~AsyncGRPCServer() {}
  void WaitServerReady() override;
  void StartServer() override;

 private:
  // HandleRequest needs to be thread-safe.
  void HandleRequest(
      ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
      std::function<void(const std::string&, int)> TryToRegisterNewOne);

  void TryToRegisterNewOne(const std::string& rpc_name, int req_id);
  void ShutdownQueue();
  void ShutDownImpl() override;

 private:
  static const int kRequestBufSize = 100;

  std::mutex cq_mutex_;
  volatile bool is_shut_down_ = false;

  GrpcService::AsyncService service_;
  std::unique_ptr<::grpc::Server> server_;

  // condition of the sub program
  std::condition_variable barrier_condition_;

  std::mutex mutex_ready_;
  std::condition_variable condition_ready_;

  int ready_;

  std::map<std::string, std::unique_ptr<::grpc::ServerCompletionQueue>> rpc_cq_;
  std::map<std::string, std::vector<std::unique_ptr<std::thread>>> rpc_threads_;
  std::map<std::string, std::vector<RequestBase*>> rpc_reqs_;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
