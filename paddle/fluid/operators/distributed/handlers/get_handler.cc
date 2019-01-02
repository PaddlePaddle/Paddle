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

#include "paddle/fluid/operators/distributed/handlers/get_handler.h"

#include <string>
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {

bool GetHandlerSync::Handle(RPCRequest *request, Scope *scope) {
  rpc_server_->WaitState(RPCServerState::STATE_RECV);
  *(request->out_var_) = scope->FindVar(request->varname_);
  return true;
}

bool GetHandlerAsync::Handle(RPCRequest *request, Scope *scope) {
  *(request->out_var_) = scope->FindVar(request->varname_);
  return true;
}

bool GetHandlerDCAsync::Handle(RPCRequest *request, Scope *scope) {
  // NOTE: the format is determined by distributed_transpiler.py
  std::string param_bak_name = string::Sprintf(
      "%s.trainer_%d_bak", request->varname_, request->trainer_id_);
  VLOG(3) << "getting " << param_bak_name << " trainer_id "
          << request->trainer_id_;
  auto var = scope->FindVar(request->varname_);
  auto t_orig = var->Get<framework::LoDTensor>();
  auto param_bak = scope_->Var(param_bak_name);
  auto t = param_bak->GetMutable<framework::LoDTensor>();
  t->mutable_data(dev_ctx_->GetPlace(), t_orig.type());
  VLOG(3) << "copying " << request->varname_ << " to " << param_bak_name;
  framework::TensorCopy(t_orig, dev_ctx_->GetPlace(), t);
  *(request->out_var_) = scope->FindVar(request->varname_);
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
