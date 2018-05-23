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
  explicit RPCProcessorCtx(
      bool sync_mode, framework::Scope* scope, platform::DeviceContext* dev_ctx,
      std::unique_ptr<framework::ExecutorPrepareContext> prepared,
      framework::ProgramDesc* program, framework::Executor* executor,
      ReceivedQueue* var_recv_queue)
      : sync_mode_(sync_mode) {
    scope_ = scope;
    dev_ctx_ = dev_ctx;

    prefetch_ctx_.reset(prepared.release());
    program_ = program;
    executor_ = executor;
    var_recv_queue_ = var_recv_queue;
  }

  bool sync_mode() { return sync_mode_; }
  framework::Scope* scope() { return scope_; }
  const platform::DeviceContext* dev_ctx() { return dev_ctx_; }
  framework::ExecutorPrepareContext* prefetch_ctx() {
    return prefetch_ctx_.get();
  }
  framework::ProgramDesc* program() { return program_; }
  framework::Executor* executor() { return executor_; }
  ReceivedQueue* var_recv_queue() { return var_recv_queue_; }

 protected:
  const bool sync_mode_;
  framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;

  std::unique_ptr<framework::ExecutorPrepareContext> prefetch_ctx_;
  framework::ProgramDesc* program_;
  framework::Executor* executor_;
  ReceivedQueue* var_recv_queue_;
};

class GRPCProcessorCtx : public RPCProcessorCtx {
 public:
  explicit GRPCProcessorCtx(
      bool sync_mode, framework::Scope* scope, platform::DeviceContext* dev_ctx,
      std::unique_ptr<framework::ExecutorPrepareContext> prepared,
      framework::ProgramDesc* program, framework::Executor* executor,
      ReceivedQueue* var_recv_queue)
      : RPCProcessorCtx(sync_mode, scope, dev_ctx, std::move(prepared), program,
                        executor, var_recv_queue) {}

  bool RequestSend(const VariableResponse* request);
  bool RequestGet(const VariableResponse* request, ::grpc::ByteBuffer* reply);
  bool ReqeustPrefetch(const VariableResponse* request,
                       ::grpc::ByteBuffer* reply);
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
