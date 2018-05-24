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

#include <iostream>
#include <string>
#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/rpc_processor.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

bool GRPCProcessorCtx::RequestSend(std::shared_ptr<VariableResponse> request) {
  var_recv_queue_.Push(std::make_pair(request->Varname(), request));
  return true;
}

bool GRPCProcessorCtx::RequestGet(const sendrecv::VariableMessage* request,
                                  ::grpc::ByteBuffer* reply) {
  auto var_name = request->varname();
  auto* var = scope_->FindVar(var_name);

  if (var_name != FETCH_BARRIER_MESSAGE) {
    SerializeToByteBuffer(var_name, var, *dev_ctx_, reply);
  }

  return true;
}

bool GRPCProcessorCtx::RequestPrefetch(const VariableResponse* request,
                                       ::grpc::ByteBuffer* reply) {
  std::string var_name = request->OutVarname();
  VLOG(3) << "RequestPrefetch " << var_name;
  auto var_desc = program_->Block(0).FindVar(var_name);
  framework::Scope* local_scope = &scope_->NewScope();
  auto* var = local_scope->FindVar(var_name);
  InitializeVariable(var, var_desc->GetType());
  executor_->RunPreparedContext(prefetch_ctx_.get(), scope_);

  SerializeToByteBuffer(var_name, var, *dev_ctx_, reply);

  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
