// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <brpc/server.h>
#include <glog/logging.h>

#include <string>

#include "paddle/fluid/distributed/rpc/python_rpc_handler.h"
#include "paddle/fluid/distributed/rpc/rpc.pb.h"

namespace paddle {
namespace distributed {
class RpcService : public RpcBaseService {
 public:
  RpcService() {}
  virtual ~RpcService() {}
  virtual void Send(google::protobuf::RpcController *cntl_base,
                    const RpcRequest *request,
                    RpcResponse *response,
                    google::protobuf::Closure *done) {
    // This object helps you to call done->Run() in RAII style. If you need
    // to process the request asynchronously, pass done_guard.release().
    brpc::ClosureGuard done_guard(done);

    brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
    LOG(INFO) << "Received request[log_id=" << cntl->log_id() << "] from "
              << cntl->remote_side() << " to " << cntl->local_side() << ": "
              << request->message()
              << " (attached=" << cntl->request_attachment() << ")";
    // Fill response.
    response->set_message(request->message());
  }

  virtual void InvokeRpc(google::protobuf::RpcController *cntl_base,
                         const RpcRequest *request,
                         RpcResponse *response,
                         google::protobuf::Closure *done) {
    brpc::ClosureGuard done_guard(done);

    brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
    VLOG(2) << "InvokeRpc API: Received request[log_id=" << cntl->log_id()
            << "] from " << cntl->remote_side() << " to " << cntl->local_side()
            << ": "
            << " (attached=" << cntl->request_attachment() << ")";
    std::string py_func_str = request->message();
    std::shared_ptr<PythonRpcHandler> python_handler =
        PythonRpcHandler::GetInstance();
    // acquire gil, because native Python objects are used
    py::gil_scoped_acquire ag;
    py::object py_func_obj = python_handler->Deserialize(py_func_str);
    py::object res = python_handler->RunPythonFunc(py_func_obj);
    std::string res_str = python_handler->Serialize(res);
    response->set_message(res_str);
  }
};
}  // namespace distributed
}  // namespace paddle
