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

#include <time.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

typedef std::pair<std::string, std::shared_ptr<VariableResponse>>
    ReceivedMessage;
typedef std::pair<std::string, sendrecv::VariableMessage> MessageWithName;
typedef framework::BlockingQueue<ReceivedMessage> ReceivedQueue;

class RPCProcessorCtx {
 public:
  RPCProcessorCtx()
      : sync_mode_(true),
        scope_(nullptr),
        dev_ctx_(nullptr),
        program_(nullptr),
        executor_(nullptr) {}
  virtual ~RPCProcessorCtx() {}

  void SetSyncMode(bool sync_mode) { sync_mode_ = sync_mode; }
  void SetScope(framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }
  void SetExecutor(framework::Executor* executor) { executor_ = executor; }
  void SetPrefetchPreparedCtx(
      std::unique_ptr<framework::ExecutorPrepareContext> prepared) {
    prefetch_ctx_.reset(prepared.release());
  }

  bool sync_mode() { return sync_mode_; }
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ExecutorPrepareContext* prefetch_ctx() {
    return prefetch_ctx_.get();
  }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }

  const ReceivedMessage Get() { return var_recv_queue_.Pop(); }

  void Push(const std::string& msg_name) {
    var_recv_queue_.Push(std::make_pair(msg_name, nullptr));
  }

 protected:
  bool sync_mode_;
  framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;

  std::unique_ptr<framework::ExecutorPrepareContext> prefetch_ctx_;
  framework::ProgramDesc* program_;
  framework::Executor* executor_;

  ReceivedQueue var_recv_queue_;
};

class GRPCProcessorCtx : public RPCProcessorCtx {
 public:
  GRPCProcessorCtx() {}
  virtual ~GRPCProcessorCtx() {}

  bool RequestSend(std::shared_ptr<VariableResponse> request);
  bool RequestGet(const sendrecv::VariableMessage* request,
                  ::grpc::ByteBuffer* reply);
  bool RequestPrefetch(const VariableResponse* request,
                       ::grpc::ByteBuffer* reply);
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
